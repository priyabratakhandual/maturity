pipeline {
    agent any

    environment {
        IMAGE_NAME = "priyabratakhandual/globtier-maturity-assessment:1.0"
        CONTAINER_NAME = "maturity-app"
        PORT = "7000"
    }

    stages {

        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/priyabratakhandual/maturity.git'
            }
        }

        stage('Pull Docker Image') {
            steps {
                sh 'docker pull $IMAGE_NAME'
            }
        }

        stage('Trivy Scan') {
            steps {
                sh '''
                trivy image --severity HIGH,CRITICAL $IMAGE_NAME
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                docker stop $CONTAINER_NAME || true
                docker rm $CONTAINER_NAME || true

                docker run -d -p $PORT:$PORT --name $CONTAINER_NAME $IMAGE_NAME
                '''
            }
        }

        stage('Wait for App') {
            steps {
                sh '''
                echo "Waiting for app to start..."

                for i in $(seq 1 20); do
                  if curl -s http://localhost:$PORT/maturity-assessments/assessment > /dev/null; then
                    echo "App is up!"
                    exit 0
                  fi
                  echo "Retry $i..."
                  sleep 3
                done

                echo "App failed to start"
                docker logs $CONTAINER_NAME
                exit 1
                '''
            }
        }

        stage('OWASP ZAP Scan') {
            steps {
                sh '''
                chmod -R 777 $(pwd)

                docker run --rm --network host \
                -v $(pwd):/zap/wrk/:rw \
                zaproxy/zap-stable zap-baseline.py \
                -t http://localhost:$PORT/maturity-assessments/assessment \
                -r zap-report.html \
                -J zap-report.json || true
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: '*.html, *.json', allowEmptyArchive: true
        }
    }
}