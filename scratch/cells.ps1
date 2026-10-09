$r = Import-Csv results/significance_runs.csv
$have = $r | Group-Object dataset, method | ForEach-Object { $_.Name + ' x' + $_.Count }
Write-Output '--- present (dataset, method x seedcount) ---'
$have | Sort-Object | ForEach-Object { Write-Output $_ }
Write-Output ('total cells rows: ' + $r.Count)
