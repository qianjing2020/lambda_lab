# settings.py

import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

stakeholder_db_user = os.environ.get('stakeholder_db_user')
stakeholder_db_password = os.environ.get('stakeholder_db_password')
stakeholder_db_host = os.environ.get('stakeholder_db_host')
stakeholder_db_name = os.environ.get('stakeholder_db_name')

    
aws_db_user = os.environ.get('aws_db_user')
aws_db_password = os.environ.get('aws_db_password')
aws_db_host = os.environ.get('aws_db_host')
aws_db_port = os.environ.get('aws_db_port')
aws_db_name = os.environ.get('aws_db_name')
