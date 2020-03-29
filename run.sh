echo 'Installing virtualenv'
pip3 install virtualenv
echo 'creating a venv'
virtualenv -p python3.7 venv
echo 'activating venv'
. ./venv/bin/activate
echo 'installing requirements.txt'
pip install -r requirements.txt
echo 'start cli ...'
python news_classifier/classifier_2.py
