# Memory-bounded significance regeneration.
# One (dataset, seed, method) per process so memory is released between cells.
# Threads capped to 1 to minimise each process's peak RSS (tight commit limit).
# Cached cells are skipped automatically by 06_significance.py.
$env:PYTHONPATH = "D:\cac-blackbox-to-action"
$env:OMP_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:NUMEXPR_NUM_THREADS = "1"
Set-Location "D:\cac-blackbox-to-action"

$log = "scratch/sig_driver.log"
# small datasets first (fast, give CIs early); adult last (heaviest, OOM-prone)
$datasets = @("german", "diabetes", "heloc", "adult")
$seeds    = @(42, 0, 1, 2, 3)
$methods  = @("random", "wachter", "dice", "oracle", "cac")

Add-Content $log ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] DRIVER START (method-level, threads=1)")
foreach ($d in $datasets) {
    foreach ($s in $seeds) {
        foreach ($m in $methods) {
            $ts = Get-Date -Format HH:mm:ss
            Add-Content $log "[$ts] START $d seed=$s method=$m"
            # cac must be in the method list to satisfy the REFERENCE check (line 353);
            # it is already cached for every dataset so it is filtered from `pending`
            # and never recomputes -- only the baseline $m actually runs and saves.
            & python experiments/06_significance.py -d $d -s $s -m $m cac -n 100 -k 5 *>> $log
            $ec = $LASTEXITCODE
            $ts2 = Get-Date -Format HH:mm:ss
            Add-Content $log "[$ts2] END   $d seed=$s method=$m (exit $ec)"
            [System.GC]::Collect()
            Start-Sleep -Milliseconds 400
        }
    }
}
# final pass: re-tabulate aggregate + tests from the per-cell CSVs (no search)
Add-Content $log ("[" + (Get-Date -Format "HH:mm:ss") + "] AGGREGATE-ONLY pass")
& python experiments/06_significance.py -d german diabetes heloc adult -s 42 0 1 2 3 --aggregate-only *>> $log
Add-Content $log ("[" + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + "] ALL DONE (exit $LASTEXITCODE)")
