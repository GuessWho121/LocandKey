param(
    [string]$AlicePort = 'COM5',
    [string]$BobPort = 'COM7',
    [string]$OutputDir = 'captures\esp32',
    [int]$DurationSeconds = 30,
    [switch]$RelayBob
)

$ErrorActionPreference = 'Stop'
$destination = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($destination) | Out-Null
$ports = @([System.IO.Ports.SerialPort]::new($AlicePort, 115200))
if (-not $RelayBob) { $ports += [System.IO.Ports.SerialPort]::new($BobPort, 115200) }
$writers = @()
$counts = @(0, 0)
$duplicates = @(0, 0)
$seen = @(
    [System.Collections.Generic.HashSet[string]]::new(),
    [System.Collections.Generic.HashSet[string]]::new()
)

try {
    foreach ($port in $ports) {
        $port.ReadTimeout = 20
        $port.Open()
    }
    $writers = @(
        [System.IO.StreamWriter]::new([System.IO.Path]::Combine($destination, 'alice.jsonl'), $false),
        [System.IO.StreamWriter]::new([System.IO.Path]::Combine($destination, 'bob.jsonl'), $false)
    )
    foreach ($writer in $writers) { $writer.AutoFlush = $true }
    Write-Host "Capturing $AlicePort$(if (-not $RelayBob) { " and $BobPort" }) at 115200 baud. Press Ctrl+C to stop."
    $deadline = if ($DurationSeconds -gt 0) { [DateTime]::UtcNow.AddSeconds($DurationSeconds) } else { [DateTime]::MaxValue }
    while ([DateTime]::UtcNow -lt $deadline) {
        for ($index = 0; $index -lt $ports.Count; $index++) {
            try { $line = $ports[$index].ReadLine().Trim() }
            catch [System.TimeoutException] { continue }
            if ($line.StartsWith('{')) {
                try {
                    $record = $line | ConvertFrom-Json
                    $sequence = [uint32]$record.sequence
                    if ($record.csi.Count -ne 128 -or $record.csi_length -ne 128) { throw 'CSI length mismatch' }
                    if ($RelayBob -and $null -eq $record.capture_id) { throw 'Missing capture ID' }
                }
                catch { Write-Host "$($ports[$index].PortName): invalid CSI record"; continue }
                $side = if ($record.device -eq 'alice') { 0 } elseif ($record.device -eq 'bob') { 1 } else { continue }
                if (-not $RelayBob -and $side -ne $index) { continue }
                $key = "$($record.capture_id):$sequence"
                if (-not $seen[$side].Add($key)) { $duplicates[$side]++; continue }
                $writers[$side].WriteLine($line)
                $counts[$side]++
                Write-Host "$($ports[$index].PortName): $($record.device) seq=$($record.sequence) RSSI=$($record.rssi_dbm) dBm RTT=$($record.round_trip_us) us CSI=$($record.csi.Count) values"
            } elseif ($line) {
                Write-Host "$($ports[$index].PortName): $line"
            }
        }
    }
} finally {
    foreach ($writer in $writers) { $writer.Dispose() }
    foreach ($port in $ports) { if ($port.IsOpen) { $port.Close() }; $port.Dispose() }
    $paired = 0
    foreach ($key in $seen[0]) { if ($seen[1].Contains($key)) { $paired++ } }
    Write-Host "Saved $($counts[0]) Alice and $($counts[1]) Bob CSI packets to $destination; paired=$paired alice_only=$($counts[0] - $paired) bob_only=$($counts[1] - $paired) duplicates=$($duplicates[0] + $duplicates[1])"
}
