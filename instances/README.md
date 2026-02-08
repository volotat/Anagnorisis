# Example: Running multiple Anagnorisis instances

This folder contains examples for running multiple Anagnorisis instances
simultaneously (e.g. for different family members or use cases).

## How to use

1. Copy one of the example files and customize it:

```bash
cp instances/example-personal.yaml instances/personal.yaml
```

2. Edit `instances/personal.yaml` with your actual folder paths.

3. Start and stop using the `-f` flag:

```bash
docker compose -f docker-compose.yaml -f instances/personal.yaml up -d
docker compose -f docker-compose.yaml -f instances/personal.yaml down
```

> **Important:** When using `-f`, `docker-compose.override.yaml` is NOT merged.
> Each instance file must contain **all** user-specific settings (ports,
> volumes, container name, etc.).

You can run multiple instances at the same time, each with different media
folders and ports. Example:

```bash
docker compose -f docker-compose.yaml -f instances/personal.yaml up -d
docker compose -f docker-compose.yaml -f instances/family.yaml up -d
```

## Instance requirements

Each instance needs:

- A unique `name` (project name at the top of the file — this is what prevents instances from overwriting each other)
- A unique `container_name`
- A unique port (the first number in the `ports` mapping)
- Its own `project_config` folder (for separate databases and trained models)

