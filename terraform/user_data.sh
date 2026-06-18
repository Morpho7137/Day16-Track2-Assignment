#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU benchmark node"

dnf update -y
dnf install -y python3 python3-pip git unzip

python3 -m pip install --upgrade pip
python3 -m pip install lightgbm scikit-learn pandas numpy kaggle

mkdir -p /home/ec2-user/ml-benchmark
chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark

echo "CPU benchmark dependencies installed"
