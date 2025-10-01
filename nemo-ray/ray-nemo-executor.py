fimport argparse
from nemo_run.core.execution.kuberay import KubeRayExecutor, KubeRayWorkerGroup
from nemo_run.run.ray.cluster import RayCluster
from nemo_run.run.ray.job import RayJob


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit a Ray job via NeMo-Run")
    parser.add_argument("--name", default="demo", help="Base name for cluster + job")
    parser.add_argument(
        "--image",
        default="anyscale/ray:2.43.0-py312-cu125",
        help="Ray container image",
    )
    parser.add_argument(
        "--command",
        default="python script.py",
        help="Entrypoint to execute inside Ray job",
    )
    args = parser.parse_args()

    # 1) Build the executor programmatically
   # 1) Configure a KubeRay executor (resources + cluster policy)
    executor = KubeRayExecutor(
       namespace="my-k8s-namespace",
       ray_version="2.43.0",
       image="anyscale/ray:2.43.0-py312-cu125",
       head_cpu="4",
       head_memory="12Gi",
       worker_groups=[
          KubeRayWorkerGroup(
            group_name="worker",        # arbitrary string
            replicas=2,                  # two worker pods
            gpus_per_worker=8,
          )
       ],
       # Optional tweaks ----------------------------------------------------
       reuse_volumes_in_worker_groups=True,          # mount PVCs on workers too
       spec_kwargs={"schedulerName": "runai-scheduler"},  # e.g. Run:ai
       volume_mounts=[{"name": "workspace", "mountPath": "/workspace"}],
       volumes=[{
          "name": "workspace",
          "persistentVolumeClaim": {"claimName": "my-workspace-pvc"},
       }],
       env_vars={
         "UV_PROJECT_ENVIRONMENT": "/home/ray/venvs/driver",
         "NEMO_RL_VENV_DIR": "/home/ray/venvs",
         "HF_HOME": "/workspace/hf_cache",
       },
       container_kwargs={
          "securityContext": {
              "allowPrivilegeEscalation": False,
              "runAsUser": 0,
           }
       },
    )

    # 2) Commands executed in EVERY Ray container before the daemon starts
    pre_ray_start = [
       "pip install uv",
       "echo 'unset RAY_RUNTIME_ENV_HOOK' >> /home/ray/.bashrc",
    ]

    # 3) Spin-up the cluster & expose the dashboard
    cluster = RayCluster(name="rick-nemo-cluster", executor=executor)
    cluster.start(timeout=900, pre_ray_start_commands=pre_ray_start)
    #cluster.port_forward(port=8265, target_port=8265, wait=False)  # dashboard → http://localhost:8265

    # 4) Submit a Ray Job that runs inside that cluster
    job = RayJob(name="demo-kuberay-job", executor=executor)
    job.start(
       command="uv run python examples/train.py --config cfgs/train.yaml",
       workdir="/path/to/project/",                # synced to PVC automatically
       runtime_env_yaml="/path/to/runtime_env.yaml",  # optional
       pre_ray_start_commands=pre_ray_start,
    )
    job.logs(follow=True)

    # 5) Clean-up
    cluster.stop()


if __name__ == "__main__":
    main()