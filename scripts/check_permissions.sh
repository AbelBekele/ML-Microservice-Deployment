#!/bin/bash

# AWS region
REGION="eu-west-1"

# Resource names
CLUSTER="automation-challenge"
REPOSITORY="automation_challenge"
LOG_GROUP="automation_challenge/abel"

# Test ECS permissions
echo "Testing ECS permissions..."
aws ecs describe-clusters --clusters $CLUSTER --region $REGION

# Test ECR permissions
echo "Testing ECR permissions..."
aws ecr describe-repositories --repository-names $REPOSITORY --region $REGION

# Test RDS permissions
echo "Testing RDS permissions..."
aws rds describe-db-instances --db-instance-identifier $CLUSTER --region $REGION

# Test S3 permissions
echo "Testing S3 permissions..."
aws s3 ls s3://automation-challenge

# Test CloudWatch Logs permissions
echo "Testing CloudWatch Logs permissions..."
aws logs describe-log-groups --log-group-name-prefix $LOG_GROUP --region $REGION

echo "Done."
