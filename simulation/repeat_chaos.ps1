while ($true) {
    Write-Host "Applying chaos experiments..."
    kubectl apply -f C:\KubernetesHackathon\pod-failure.yaml
    kubectl apply -f C:\KubernetesHackathon\stress-chaos.yaml
    kubectl apply -f C:\KubernetesHackathon\network-chaos.yaml
    kubectl apply -f C:\KubernetesHackathon\service-disruption.yaml
    # Uncomment the next line if you want to include the scheduled pod failure
    # kubectl apply -f C:\KubernetesHackathon\pod-failure-schedule.yaml
    Write-Host "Waiting 5 minutes for chaos to run..."
    Start-Sleep -Seconds 300  # Wait 5 minutes (matches longest duration)
    Write-Host "Deleting chaos experiments..."
    kubectl delete -f C:\KubernetesHackathon\pod-failure.yaml
    kubectl delete -f C:\KubernetesHackathon\stress-chaos.yaml
    kubectl delete -f C:\KubernetesHackathon\network-chaos.yaml
    kubectl delete -f C:\KubernetesHackathon\service-disruption.yaml
    # Uncomment the next line if you included pod-failure-schedule.yaml
    # kubectl delete -f C:\KubernetesHackathon\pod-failure-schedule.yaml
    Write-Host "Chaos cycle completed. Starting next cycle in 10 seconds..."
    Start-Sleep -Seconds 10  # Wait 10 seconds before the next cycle
}