Set-Location "D:\cac-blackbox-to-action"
if (Test-Path scratch/sig_driver.log) { Move-Item scratch/sig_driver.log scratch/sig_driver.run2old.log -Force }
$p = Start-Process -FilePath "powershell" -ArgumentList "-ExecutionPolicy","Bypass","-File","D:\cac-blackbox-to-action\scratch\sig_driver.ps1" -WindowStyle Hidden -PassThru
Write-Output ("LAUNCHED PID=" + $p.Id)
