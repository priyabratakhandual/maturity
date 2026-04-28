pipeline {
    agent { label 'worker2' }


    stages {
        stage('Check Node') {
            steps {
                sh 'hostname'
            }
        }

    stages {
        stage('Deploy Application') {
            steps {
                sh '''
                cd /opt/devops-project
                ansible-playbook -i inventory.ini deploy.yml
                '''
            }
        }
    }
}