pipeline {
    agent any

    environment {
        IMAGE_NAME = "priyabratakhandual/globtier-maturity-assessment"
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

        stage('OWASP ZAP Scan') {
            steps {
                sh '''
                docker run --rm --network host \
                -v $(pwd):/zap/wrk/:rw \
                owasp/zap2docker-stable zap-baseline.py \
                -t http://localhost:$PORT \
                -r zap-report.html
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'zap-report.html', allowEmptyArchive: true
        }
    }
}