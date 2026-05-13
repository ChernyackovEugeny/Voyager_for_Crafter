"""Manual smoke test for the upstream crafter package.

Kept out of unittest discovery side effects: importing this module must not
create an environment, print frames, or block on input().
"""


def main() -> None:
    import crafter

    raw_env = crafter.Env()
    obs = raw_env.reset()
    action = raw_env.action_space.sample()
    obs, reward, done, info = raw_env.step(action)
    terminated = bool(done)
    truncated = False

    print("obs_shape", getattr(obs, "shape", None))
    print("reward", reward)
    print("terminated", terminated)
    print("truncated", truncated)
    print("inventory", info.get("inventory", {}))


if __name__ == "__main__":
    main()
