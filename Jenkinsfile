pipeline {
    agent { label 'worker2' }

    stages {
        stage('Check Node') {
            steps {
                sh 'hostname'
            }
        }

        stage('Deploy Application') {
            steps {
                sh '''
                ansible-playbook -i inventory.ini deploy.yml
                '''
            }
        }
    }
}