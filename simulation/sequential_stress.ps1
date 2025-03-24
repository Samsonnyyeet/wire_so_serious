$DURATION = "300s"

function Apply-Stress {
    param (
        [int]$cpuLoad,
        [string]$memorySize
    )

    $name = "sequential-stress-$cpuLoad-$memorySize"

    $yaml = @"
apiVersion: chaos-mesh.org/v1alpha1
kind: StressChaos
metadata:
  name: $name
  namespace: default
spec:
  selector:
    labelSelectors:
      app: nginx
  mode: all
  stressors:
    memory:
      workers: 2
      size: "$memorySize"
    cpu:
      workers: 2
      load: $cpuLoad
  duration: "$DURATION"
"@

    $yaml | kubectl apply -f -
    Write-Host "Applied stress: CPU load $cpuLoad%, Memory $memorySize"
    python log_stress.py $cpuLoad $memorySize
    Start-Sleep -Seconds 300
    kubectl delete stresschaos $name -n default
    Write-Host "Deleted stress: CPU load $cpuLoad%, Memory $memorySize"
    Start-Sleep -Seconds 60
}

for ($cycle = 1; $cycle -le 3; $cycle++) {
    Write-Host "Starting cycle $cycle..."
    Apply-Stress -cpuLoad 20 -memorySize "100MB"
    Apply-Stress -cpuLoad 50 -memorySize "300MB"
    Apply-Stress -cpuLoad 100 -memorySize "500MB"
}

Write-Host "Sequential stress simulation completed."