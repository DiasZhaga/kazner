// kazner pipeline for the local Jenkins in ci/jenkins (see docs/ci-jenkins.md).
// Every stage runs inside images built from the repository's Dockerfile, so the Jenkins
// controller needs only the Docker CLI and kubectl, not Python.

pipeline {
  agent any

  options {
    timestamps()
    timeout(time: 30, unit: 'MINUTES')
    buildDiscarder(logRotator(numToKeepStr: '20'))
    disableConcurrentBuilds()
  }

  environment {
    SHORT_SHA = "${env.GIT_COMMIT.take(7)}"
    TEST_IMAGE = "kazner:test-${env.GIT_COMMIT.take(7)}"
    RUNTIME_IMAGE = "kazner:${env.GIT_COMMIT.take(7)}"
    // Docker names allow only [a-zA-Z0-9_.-]; BUILD_TAG contains the branch name (ci%2F54-...).
    TEST_CONTAINER = "kazner-tests-${env.GIT_COMMIT.take(7)}-${env.BUILD_NUMBER}"
    EXAMPLE = '/opt/kazner/examples/contract_example.csv'
  }

  stages {
    stage('Build test image') {
      steps {
        sh 'docker build --target test -t "$TEST_IMAGE" .'
      }
    }

    stage('Lint') {
      steps {
        sh 'docker run --rm "$TEST_IMAGE" ruff check .'
        sh 'docker run --rm "$TEST_IMAGE" ruff format --check .'
      }
    }

    stage('Unit tests') {
      steps {
        // The controller talks to the host engine, so a bind mount would point at a host path;
        // the reports are copied out of the stopped container instead.
        sh '''
          docker rm -f "$TEST_CONTAINER" >/dev/null 2>&1 || true
          set +e
          docker run --name "$TEST_CONTAINER" "$TEST_IMAGE" python -m pytest \
            -m "not network and not smoke" -q \
            --junitxml=/tmp/reports/junit.xml \
            --cov=kazner --cov-report=term --cov-report=xml:/tmp/reports/coverage.xml
          status=$?
          set -e
          rm -rf reports
          docker cp "$TEST_CONTAINER:/tmp/reports" reports
          docker rm "$TEST_CONTAINER" >/dev/null
          exit $status
        '''
      }
      post {
        always {
          junit testResults: 'reports/junit.xml', allowEmptyResults: false
          recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'reports/coverage.xml']],
                         sourceDirectories: [[path: 'src']])
        }
      }
    }

    stage('Build runtime image') {
      steps {
        sh 'docker build -t "$RUNTIME_IMAGE" .'
        sh 'docker image ls kazner --format "{{.Repository}}:{{.Tag}}  {{.Size}}"'
      }
    }

    stage('Smoke test') {
      steps {
        sh '''
          docker run --rm "$RUNTIME_IMAGE" --version
          docker run --rm "$RUNTIME_IMAGE" evaluate --predictions "$EXAMPLE" | tee smoke.json
          grep -q '"f1": 0.8,' smoke.json
        '''
      }
    }

    stage('Push to Docker Hub') {
      when {
        allOf {
          branch 'main'
          expression { env.DOCKERHUB_REPO?.trim() }
        }
      }
      steps {
        withCredentials([usernamePassword(credentialsId: 'dockerhub',
                                          usernameVariable: 'HUB_USER',
                                          passwordVariable: 'HUB_TOKEN')]) {
          sh '''
            echo "$HUB_TOKEN" | docker login -u "$HUB_USER" --password-stdin
            docker tag "$RUNTIME_IMAGE" "$DOCKERHUB_REPO:$SHORT_SHA"
            docker tag "$RUNTIME_IMAGE" "$DOCKERHUB_REPO:latest"
            docker push "$DOCKERHUB_REPO:$SHORT_SHA"
            docker push "$DOCKERHUB_REPO:latest"
            docker logout
          '''
        }
      }
    }

    stage('Deploy to Kubernetes (test)') {
      when { branch 'main' }
      steps {
        withCredentials([file(credentialsId: 'kubeconfig-kazner-test', variable: 'KUBECONFIG')]) {
          sh '''
            JOB="kazner-evaluate-$SHORT_SHA-$BUILD_NUMBER"
            sed -e "s|__JOB__|$JOB|" -e "s|__IMAGE__|$RUNTIME_IMAGE|" deploy/k8s/evaluate-job.yaml \
              | kubectl apply -f -
            kubectl -n kazner-test wait --for=condition=complete "job/$JOB" --timeout=180s
            kubectl -n kazner-test logs "job/$JOB" | tee deploy.json
            grep -q '"f1": 0.8,' deploy.json
          '''
        }
      }
    }
  }

  post {
    always {
      archiveArtifacts artifacts: 'reports/*.xml, smoke.json, deploy.json', allowEmptyArchive: true
      // Keep the cached layers; drop this build's test image tag.
      sh 'docker rmi "$TEST_IMAGE" >/dev/null 2>&1 || true'
      cleanWs()
    }
  }
}
