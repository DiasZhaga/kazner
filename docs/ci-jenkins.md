# Self-hosted Jenkins pipeline (#54)

GitHub Actions (`.github/workflows/`) is the merge gate. A local Jenkins builds every commit
into a container image and deploys tested `main` builds to a Kubernetes test namespace. The
server itself is code:

| File | Role |
|---|---|
| `Jenkinsfile` | declarative pipeline (below) |
| `ci/jenkins/Dockerfile` | controller image: Jenkins 2.580.1 LTS (JDK 21), Docker CLI 28.4, kubectl 1.34.1 (checksum-verified) |
| `ci/jenkins/plugins.txt` | pinned top-level plugins |
| `ci/jenkins/casc.yaml` | Configuration-as-Code: admin user, one executor, credentials, the `kazner` multibranch job |
| `ci/jenkins/compose.yaml` | runs the controller with the host Docker socket and the kubeconfig secret |
| `ci/jenkins/k8s-setup.sh` | creates the test namespace and writes the deployer kubeconfig |
| `deploy/k8s/rbac.yaml`, `deploy/k8s/evaluate-job.yaml` | namespace, ServiceAccount, Role; the deployed Job |

## Pipeline

```
checkout → build test image → lint → unit tests → build runtime image → smoke test
         → push to Docker Hub (main, optional) → deploy to Kubernetes (main)
```

| Stage | What it does |
|---|---|
| Build test image | `docker build --target test`, tagged `kazner:test-<sha>` |
| Lint | `ruff check .` and `ruff format --check .` in that image |
| Unit tests | offline suite in the image; JUnit and Cobertura reports copied out with `docker cp` and published (test trend, coverage) |
| Build runtime image | `docker build`, tagged `kazner:<sha>` |
| Smoke test | `kazner --version`; `kazner evaluate` on the shipped synthetic example must print `"f1": 0.8,` |
| Push to Docker Hub | only on `main` and only when `DOCKERHUB_REPO` is set; tags `<sha>` and `latest` |
| Deploy to Kubernetes (test) | only on `main`: a Job `kazner-evaluate-<sha>-<build>` in `kazner-test` runs the same evaluation in the cluster; the stage waits for completion and checks the log |

Every branch with a `Jenkinsfile` is built. The job polls GitHub every 2 minutes, because a
webhook cannot reach a Jenkins on localhost. When Jenkins is first started, the branches that
already exist are not built, so that ~30 old branches do not build at once. The same strategy
also skips the first discovery of a new branch: start that first build by hand, or push a
second commit. After that, every commit is built.

## Start

Requirements: Docker Desktop with **Kubernetes enabled (kubeadm)**. With kubeadm, the cluster
uses the same engine as `docker build`, so `imagePullPolicy: IfNotPresent` finds the images
without a registry.

```bash
cp ci/jenkins/.env.example ci/jenkins/.env        # set JENKINS_ADMIN_PASSWORD
bash ci/jenkins/k8s-setup.sh                      # namespace + ci/jenkins/secrets/kubeconfig
docker compose -f ci/jenkins/compose.yaml up -d --build
```

Open <http://localhost:8080> and log in as `admin`. The `kazner` job appears. Its first scan
lists the branches; trigger a build of `main` by hand.

Without Kubernetes, create an empty `ci/jenkins/secrets/kubeconfig` instead of running the
setup script. Then everything except the deploy stage works.

## Optional: Docker Hub

1. On Docker Hub create the repository and an access token (*Account settings → Personal
   access tokens*, read & write).
2. In Jenkins: *Manage Jenkins → Credentials → System → Global → Add credentials*, kind
   *Username with password*, ID **`dockerhub`**, username = Docker Hub user, password = the
   token.
3. Set `DOCKERHUB_REPO=<user>/kazner` in `ci/jenkins/.env` and restart:
   `docker compose -f ci/jenkins/compose.yaml up -d`.

## Security notes

- The deployer account can create and read Jobs and read pods and logs in `kazner-test`
  only (`kubectl auth can-i create jobs -n default` → `no`). Its token is in
  `ci/jenkins/secrets/` (ignored by git) and reaches Jenkins as a Docker secret and a
  Credentials entry, never as an environment variable in a log.
- Jenkins is bound to `127.0.0.1`. The controller gets the Docker socket through group 0
  instead of running as root, but access to the socket is still root-equivalent on the
  Docker VM. This is acceptable for a single-user laptop, not for a shared server, where
  builds would run on separate agents.
- Builds run on the controller's single executor. A production setup would use agents.

## Jenkins and GitHub Actions

| | GitHub Actions (`ci.yml`) | Jenkins (`Jenkinsfile`) |
|---|---|---|
| Role | merge gate (branch protection) | container build and test deployment |
| Runs on | hosted Ubuntu + Windows runners | the laptop's Docker Desktop |
| Environment | fresh VM per job | the repository's Docker images |
| Deploy | releases on `v*` tags | Kubernetes Job in `kazner-test` |
| Trigger | push / pull request events | polling every 2 minutes |
