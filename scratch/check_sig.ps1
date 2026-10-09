Write-Output ('runs: ' + (Import-Csv results/significance_runs.csv).Count + ' / 100')
$r = Import-Csv results/significance_runs.csv
$r | Group-Object dataset | ForEach-Object { Write-Output ($_.Name + ': ' + $_.Count) }
$d = Get-ChildItem scratch/sig_driver*.log -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($d) {
  Write-Output ('newest log: ' + $d.Name + '  (modified ' + $d.LastWriteTime + ')')
  $done = Select-String -Path $d.FullName -Pattern 'ALL DONE' -Encoding Unicode
  Write-Output ('ALL DONE: ' + [bool]$done)
}
Write-Output '--- result CSV timestamps ---'
Get-ChildItem results/significance_*.csv | ForEach-Object { Write-Output ($_.Name + '  ' + $_.LastWriteTime) }
