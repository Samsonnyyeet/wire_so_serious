$DURATION = "300s"

function Apply-RandomStress {
    # Random CPU load between 10% and 100%
    $cpuLoad = Get-Random -Minimum 10 -Maximum 101
    # Random memory size between 50MB and 500MB
    $memorySize = Get-Random -Minimum 50 -Maximum 501
    $memorySizeStr = "${memorySize}MB"
    # Random name to avoid conflicts
    $name = "random-stress-$(Get-Date -Format 'yyyyMMddHHmmss')"

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
      size: "$memorySizeStr"
    cpu:
      workers: 2
      load: $cpuLoad
  duration: "$DURATION"
"@

    $yaml | kubectl apply -f -
    Write-Host "Applied random stress: CPU load $cpuLoad%, Memory $memorySizeStr"
    python log_stress.py $cpuLoad $memorySizeStr
    Start-Sleep -Seconds 300
    kubectl delete stresschaos $name -n default
    Write-Host "Deleted random stress: CPU load $cpuLoad%, Memory $memorySizeStr"
    # Random wait between 30 and 120 seconds
    $waitTime = Get-Random -Minimum 30 -Maximum 121
    Start-Sleep -Seconds $waitTime
}

# Run for approximately 1 hour (12 iterations of 5-minute stress + random waits)
for ($i = 1; $i -le 12; $i++) {
    Write-Host "Random stress iteration $i..."
    Apply-RandomStress
}

Write-Host "Random stress simulation completed."