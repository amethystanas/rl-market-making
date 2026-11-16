"""Train the A2C (Advantage Actor-Critic) market-making agent.

    python -m agents.rl.train_a2c --data data/SOLUSDT-aggTrades-2026-02.csv

Algorithm (one update per episode of ``episode_length`` steps):

1. Roll out the stochastic policy π_θ and store (s_t, a_t, r_t).
2. Bootstrapped returns:   R_t = r_t + γ R_{t+1},   R_T = V_φ(s_T)
3. Advantages:             A_t = R_t − V_φ(s_t)      (normalised per batch)
4. Critic loss:            L_V = mean (R_t − V_φ(s_t))²
5. Actor loss:             L_π = −mean [ log π_θ(a_t|s_t) · A_t ] − β · H[π_θ(·|s_t)]

The data is split chronologically into train / validation / test. The
checkpoint with the best greedy return on the validation split is kept; the
test split is only used by ``agents.rl.evaluate``.

References:
  * Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016
  * https://huggingface.co/learn/deep-rl-course/unit6/advantage-actor-critic
  * https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.rl.backtest import actor_policy, run_backtest
from agents.rl.env import MM_ACTION_NAMES, EnvConfig, MarketMakingEnv
from agents.rl.features import BarSeries, load_agg_trades
from agents.rl.networks import Actor, Critic, save_checkpoint

TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15


@dataclass
class TrainConfig:
    episodes: int = 2000
    gamma: float = 0.99
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    entropy_start: float = 0.02
    entropy_end: float = 0.001
    max_grad_norm: float = 0.5
    eval_every: int = 100
    mirror_augment: bool = True
    seed: int = 0


def split_bars(bars: BarSeries) -> tuple[BarSeries, BarSeries, BarSeries]:
    """Chronological train / validation / test split (no shuffling: no look-ahead)."""
    n = len(bars)
    a, b = int(n * TRAIN_FRACTION), int(n * (TRAIN_FRACTION + VAL_FRACTION))
    return bars.slice(0, a), bars.slice(a, b), bars.slice(b, n)


def discounted_returns(rewards: list[float], bootstrap: float, gamma: float) -> np.ndarray:
    returns = np.empty(len(rewards), dtype=np.float32)
    running = bootstrap
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        returns[t] = running
    return returns


def train(bars: BarSeries, env_cfg: EnvConfig, cfg: TrainConfig, volume_threshold: float,
          out_path: Path, log_path: Path) -> list[dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(1)          # small networks: one thread is fastest

    train_bars, val_bars, _ = split_bars(bars)
    envs = [MarketMakingEnv(train_bars, env_cfg, seed=cfg.seed)]
    if cfg.mirror_augment:
        envs.append(MarketMakingEnv(train_bars.mirrored(), env_cfg, seed=cfg.seed + 1))
    env = envs[0]
    actor, critic = Actor(env.obs_size), Critic(env.obs_size)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr_actor)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic)

    print(f"[A2C] bars: train={len(train_bars)} val={len(val_bars)} | "
          f"obs={env.obs_size} actions={env.n_actions}")

    history, best_val, start = [], -np.inf, time.time()
    for episode in range(1, cfg.episodes + 1):
        # ---- 1. rollout -----------------------------------------------------
        env = envs[np.random.randint(len(envs))]
        obs, done = env.reset(), False
        states, actions, rewards = [], [], []
        while not done:
            with torch.no_grad():
                action = int(actor.distribution(torch.as_tensor(obs).unsqueeze(0)).sample())
            next_obs, reward, done, _ = env.step(action)
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        states_t = torch.as_tensor(np.array(states))
        actions_t = torch.as_tensor(actions)

        # ---- 2-3. returns and advantages ----------------------------------
        with torch.no_grad():
            bootstrap = float(critic(torch.as_tensor(obs).unsqueeze(0)))
            returns_t = torch.as_tensor(discounted_returns(rewards, bootstrap, cfg.gamma))
            advantages = returns_t - critic(states_t)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 4. critic update ---------------------------------------------
        critic_loss = F.mse_loss(critic(states_t), returns_t)
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        critic_opt.step()

        # ---- 5. actor update (linearly decaying entropy bonus) -------------
        progress = (episode - 1) / max(1, cfg.episodes - 1)
        beta = cfg.entropy_start + progress * (cfg.entropy_end - cfg.entropy_start)
        dist = actor.distribution(states_t)
        entropy = dist.entropy().mean()
        actor_loss = -(dist.log_prob(actions_t) * advantages).mean() - beta * entropy
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        actor_opt.step()

        # ---- logging / validation -----------------------------------------
        counts = np.bincount(actions, minlength=3)
        row = {"episode": episode, "reward": float(np.sum(rewards)),
               "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
               "entropy": entropy.item(),
               **{name.lower(): int(c) for name, c in zip(MM_ACTION_NAMES, counts)},
               "val_return": ""}

        if episode % cfg.eval_every == 0 or episode == cfg.episodes:
            actor.eval()
            val = run_backtest("val", val_bars, env_cfg, actor_policy(actor))
            actor.train()
            row["val_return"] = val.total_return
            tag = ""
            if val.total_return > best_val:
                best_val = val.total_return
                save_checkpoint(out_path, actor, critic, env_cfg, volume_threshold,
                                extra={"episode": episode, "val_return": best_val,
                                       "train_config": vars(cfg)})
                tag = "  <- best, saved"
            recent = history[-cfg.eval_every + 1:] + [row]
            print(f"ep {episode:5d} | reward {np.mean([r['reward'] for r in recent]):+7.2f} | "
                  f"entropy {entropy.item():.3f} | N/LS/LB {counts[0]:3d}/{counts[1]:3d}/{counts[2]:3d} | "
                  f"val return {val.total_return:+6.2f}% ({val.fills} fills) | "
                  f"{time.time() - start:5.0f}s{tag}")
        history.append(row)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"[A2C] best validation return {best_val:+.2f}% -> {out_path}")
    return history


def plot_training(history: list[dict], path: Path, smooth: int = 50) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def series(key):
        return np.array([h[key] for h in history], dtype=float)

    def moving_average(x):
        k = min(smooth, len(x))
        return np.convolve(x, np.ones(k) / k, mode="valid"), np.arange(k - 1, len(x))

    episodes = series("episode")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    panels = [("reward", "Episode reward (train)", "tab:blue"),
              ("critic_loss", "Critic loss  MSE(R, V(s))", "tab:red"),
              ("actor_loss", "Actor loss  −log π · A − βH", "tab:green"),
              ("entropy", "Policy entropy", "tab:purple")]
    for ax, (key, title, color) in zip(axes.flat[[0, 2, 3, 4]], panels):
        y = series(key)
        ma, x = moving_average(y)
        ax.plot(episodes, y, color=color, alpha=0.2)
        ax.plot(episodes[x], ma, color=color, lw=2)
        ax.set_title(title)
    axes[0, 2].set_yscale("log")

    evals = [h for h in history if h["val_return"] != ""]
    ax = axes[0, 1]
    ax.plot([h["episode"] for h in evals], [h["val_return"] for h in evals], "o-", color="tab:orange")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("Validation return, greedy policy (%)")

    ax = axes[1, 2]
    for name, color in zip(MM_ACTION_NAMES, ("tab:gray", "tab:red", "tab:green")):
        ma, x = moving_average(series(name.lower()))
        ax.plot(episodes[x], ma, color=color, lw=2, label=name)
    ax.set_title("Actions per episode")
    ax.legend()

    for ax in axes.flat:
        ax.set_xlabel("episode")
        ax.grid(alpha=0.3)
    fig.suptitle("A2C market maker: training", fontsize=15)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[A2C] training curves -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data", default="data/SOLUSDT-aggTrades-2026-02.csv")
    p.add_argument("--volume-threshold", type=float, default=1000)
    p.add_argument("--episodes", type=int, default=TrainConfig.episodes)
    p.add_argument("--cost", type=float, default=EnvConfig.cost, help="fee per fill (fraction)")
    p.add_argument("--inventory-penalty", type=float, default=EnvConfig.inventory_penalty)
    p.add_argument("--no-mirror", action="store_true", help="disable mirror augmentation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/a2c.pt")
    p.add_argument("--log", default="models/a2c_training_log.csv")
    p.add_argument("--figure", default="docs/figures/a2c_training.png")
    args = p.parse_args()

    bars = load_agg_trades(args.data, args.volume_threshold)
    env_cfg = EnvConfig(cost=args.cost, inventory_penalty=args.inventory_penalty)
    cfg = TrainConfig(episodes=args.episodes, seed=args.seed, mirror_augment=not args.no_mirror)
    history = train(bars, env_cfg, cfg, args.volume_threshold, Path(args.out), Path(args.log))
    plot_training(history, Path(args.figure))


if __name__ == "__main__":
    main()
