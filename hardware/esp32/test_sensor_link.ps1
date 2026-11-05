param(
    [switch]$Provision,
    [switch]$PowerCycleCheck,
    [switch]$Live,
    [string]$HotspotSsid = '',
    [string]$AlicePort = 'COM5',
    [string]$BobPort = 'COM7',
    [string]$OutputDir = 'captures\esp32\sensor-link-test'
)

$ErrorActionPreference = 'Stop'
if ($HotspotSsid -match "[`t`r`n]" -or $HotspotSsid.Length -gt 32) {
    throw 'Hotspot SSID must be at most 32 characters without tabs or newlines'
}
$hotspotPassword = $null
if ($HotspotSsid) {
    $securePassword = Read-Host 'Hotspot password' -AsSecureString
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)
    try { $hotspotPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer) }
    if ($hotspotPassword.Length -lt 8 -or $hotspotPassword.Length -gt 63 -or
        $hotspotPassword -match "[`t`r`n]") { throw 'Invalid hotspot password length or characters' }
}
$destination = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($destination) | Out-Null
$ports = @{
    alice = [System.IO.Ports.SerialPort]::new($AlicePort, 115200)
    bob = [System.IO.Ports.SerialPort]::new($BobPort, 115200)
}
$writers = @{}
$buffers = @{alice = ''; bob = ''}
$acceptedIds = [System.Collections.Generic.HashSet[string]]::new()
$state = @{
    aliceReady = 0
    bobReady = 0
    aliceStatus = 0
    bobStatus = 0
    done = 0
    accepted = 0
    duplicateAccepted = 0
    replays = 0
    tampers = 0
    lastBoot = 0L
    lastSequence = 0L
    bobStoredBoot = 0L
    bobStoredSequence = 0L
    aliceWifi = 0
    bobWifi = 0
    aliceChannel = 0
    bobChannel = 0
    aliceIp = ''
    bobIp = ''
}

function Read-Events {
    foreach ($side in @('alice', 'bob')) {
        $buffers[$side] += $ports[$side].ReadExisting()
        while (($ending = $buffers[$side].IndexOf("`n")) -ge 0) {
            $line = $buffers[$side].Substring(0, $ending).Trim()
            $buffers[$side] = $buffers[$side].Substring($ending + 1)
            if (-not $line) { continue }
            $writers[$side].WriteLine($line)
            if ($Live) { Write-Host "${side}: $line" }
            if ($line -match '^READY role=alice ') { $state.aliceReady++ }
            if ($line -match '^READY role=bob ') { $state.bobReady++ }
            if ($line -match '^WIFI_CONNECTED role=(alice|bob) channel=(\d+) ip=([0-9.]+)$') {
                $state["${side}Wifi"]++
                $state["${side}Channel"] = [int]$Matches[2]
                $state["${side}Ip"] = $Matches[3]
            }
            if ($line -match '^STATUS role=alice ') { $state.aliceStatus++ }
            if ($side -eq 'bob' -and $line -match '^STATUS role=bob boot=(\d+) last_seq=(\d+) ') {
                $state.bobStatus++
                $state.bobStoredBoot = [long]$Matches[1]
                $state.bobStoredSequence = [long]$Matches[2]
            }
            if ($line -eq 'DONE sending') { $state.done++ }
            if ($side -eq 'bob' -and $line -match '^ACCEPT boot=(\d+) seq=(\d+) ') {
                $state.accepted++
                if (-not $acceptedIds.Add("$($Matches[1]):$($Matches[2])")) { $state.duplicateAccepted++ }
                $state.lastBoot = [long]$Matches[1]
                $state.lastSequence = [long]$Matches[2]
            }
            if ($side -eq 'bob' -and $line -match '^REJECT replay ') { $state.replays++ }
            if ($side -eq 'bob' -and $line -match '^REJECT tamper ') { $state.tampers++ }
            if ($line -match '^(ERROR|PROVISION_INVALID|WIFI_FAILED)') { throw "$side reported $line" }
        }
    }
}

function Wait-For([scriptblock]$condition, [string]$label, [int]$seconds) {
    $deadline = [DateTime]::UtcNow.AddSeconds($seconds)
    while ([DateTime]::UtcNow -lt $deadline) {
        Read-Events
        if (& $condition) { return }
        Start-Sleep -Milliseconds 25
    }
    throw "Timed out waiting for $label; accepted=$($state.accepted), replays=$($state.replays), tampers=$($state.tampers)"
}

function Connect-Hotspot([string]$side) {
    $beforeConnection = $state["${side}Wifi"]
    $ports[$side].WriteLine("wifi`t$HotspotSsid`t$hotspotPassword")
    Wait-For { $state["${side}Wifi"] -gt $beforeConnection } "$side hotspot connection" 30
}

try {
    foreach ($side in @('alice', 'bob')) {
        $ports[$side].ReadTimeout = 100
        $ports[$side].WriteTimeout = 1000
        $ports[$side].Open()
        $writers[$side] = [System.IO.StreamWriter]::new(
            [System.IO.Path]::Combine($destination, "$side.log"), $false)
        $writers[$side].AutoFlush = $true
    }
    Start-Sleep -Seconds 2
    Read-Events
    if ($Provision) {
        $secret = [byte[]]::new(32)
        $random = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        $random.GetBytes($secret)
        $random.Dispose()
        $secretHex = [System.BitConverter]::ToString($secret).Replace('-', '')
        $ports.bob.WriteLine($secretHex)
        $ports.alice.WriteLine($secretHex)
        [System.Array]::Clear($secret, 0, $secret.Length)
        $secretHex = $null
        Wait-For { $state.aliceReady -gt 0 -and $state.bobReady -gt 0 } 'both provisioned boards' 20
    } else {
        $ports.bob.WriteLine('status')
        $ports.alice.WriteLine('status')
        Wait-For { $state.aliceStatus -gt 0 -and $state.bobStatus -gt 0 } 'both board statuses' 15
    }
    if ($HotspotSsid) {
        Connect-Hotspot 'bob'
        Connect-Hotspot 'alice'
        if ($state.aliceChannel -ne $state.bobChannel) { throw 'ESP32 hotspot channels differ' }
    }
    $physicalCycleObserved = $false
    $storedStateRestored = $false
    $powerCycleReplayRejected = $false
    if ($PowerCycleCheck) {
        if ($Provision) { throw 'PowerCycleCheck requires an already provisioned pair' }
        $beforeAccepted = $state.accepted
        $ports.alice.WriteLine('once')
        Wait-For { $state.accepted -gt $beforeAccepted } 'pre-cycle reading' 10
        $preCycleBoot = $state.lastBoot
        $preCycleSequence = $state.lastSequence
        $ports.bob.Close()
        Write-Host "POWER_CYCLE_BOB_NOW: Unplug and reconnect $BobPort at the computer end; Alice must stay connected."
        $deadline = [DateTime]::UtcNow.AddSeconds(180)
        while ([DateTime]::UtcNow -lt $deadline -and
            @([System.IO.Ports.SerialPort]::GetPortNames()) -contains $BobPort) {
            Start-Sleep -Milliseconds 250
        }
        if (@([System.IO.Ports.SerialPort]::GetPortNames()) -contains $BobPort) {
            throw "$BobPort did not disconnect within 180 seconds"
        }
        $physicalCycleObserved = $true
        Write-Host "$BobPort disconnected; waiting for reconnection."
        $deadline = [DateTime]::UtcNow.AddSeconds(180)
        while ([DateTime]::UtcNow -lt $deadline -and
            @([System.IO.Ports.SerialPort]::GetPortNames()) -notcontains $BobPort) {
            Start-Sleep -Milliseconds 250
        }
        if (@([System.IO.Ports.SerialPort]::GetPortNames()) -notcontains $BobPort) {
            throw "$BobPort did not reconnect within 180 seconds"
        }
        Start-Sleep -Seconds 2
        $ports.bob.Open()
        $buffers.bob = ''
        Read-Events
        $beforeStatus = $state.bobStatus
        $ports.bob.WriteLine('status')
        Wait-For { $state.bobStatus -gt $beforeStatus } 'Bob persisted replay status' 15
        $storedStateRestored = $state.bobStoredBoot -eq $preCycleBoot -and
            $state.bobStoredSequence -eq $preCycleSequence
        if (-not $storedStateRestored) { throw 'Bob replay high-water was not restored after power cycle' }
        if ($HotspotSsid) { Connect-Hotspot 'bob' }
        $beforeReplay = $state.replays
        $ports.alice.WriteLine('replay')
        Wait-For { $state.replays -gt $beforeReplay } 'replay rejection after power cycle' 10
        $powerCycleReplayRejected = $true
    }

    $ports.alice.WriteLine('start')
    Wait-For { $state.done -gt 0 } '120 transmitted readings' 100
    Wait-For { $state.accepted -ge 100 -and $state.lastSequence -ge 120 } '100 accepted readings and last frame' 10
    $beforeReplay = $state.replays
    $ports.alice.WriteLine('replay')
    Wait-For { $state.replays -gt $beforeReplay } 'replay rejection' 10
    $ports.alice.WriteLine('tamper')
    Wait-For { $state.tampers -gt 0 } 'tamper rejection' 10

    $beforeReady = $state.bobReady
    $ports.bob.WriteLine('reboot')
    Wait-For { $state.bobReady -gt $beforeReady } 'Bob reboot' 20
    if ($HotspotSsid) { Connect-Hotspot 'bob' }
    $beforeReplay = $state.replays
    $ports.alice.WriteLine('replay')
    Wait-For { $state.replays -gt $beforeReplay } 'replay rejection after Bob reboot' 10

    $priorBoot = $state.lastBoot
    $beforeReady = $state.aliceReady
    $ports.alice.WriteLine('reboot')
    Wait-For { $state.aliceReady -gt $beforeReady } 'Alice reboot' 20
    if ($HotspotSsid) { Connect-Hotspot 'alice' }
    $beforeAccepted = $state.accepted
    $ports.alice.WriteLine('once')
    Wait-For { $state.accepted -gt $beforeAccepted -and $state.lastBoot -gt $priorBoot } 'new boot reading' 10

    $summary = [ordered]@{
        accepted_readings = $state.accepted
        duplicate_accepted = $state.duplicateAccepted
        rejected_replays = $state.replays
        rejected_tamper = $state.tampers
        physical_power_cycle_observed = $physicalCycleObserved
        stored_state_restored = $storedStateRestored
        power_cycle_replay_rejected = $powerCycleReplayRejected
        sender_boot_advanced = $state.lastBoot -gt $priorBoot
        hotspot_ssid = $HotspotSsid
        hotspot_channel = $state.aliceChannel
        alice_ip = $state.aliceIp
        bob_ip = $state.bobIp
        passed = $state.accepted -ge 101 -and $state.duplicateAccepted -eq 0 -and
            $state.replays -ge $(if ($PowerCycleCheck) { 3 } else { 2 }) -and $state.tampers -ge 1 -and
            (-not $PowerCycleCheck -or ($physicalCycleObserved -and $storedStateRestored -and $powerCycleReplayRejected)) -and
            (-not $HotspotSsid -or ($state.aliceWifi -gt 0 -and $state.bobWifi -gt 0 -and
                $state.aliceChannel -eq $state.bobChannel -and $state.aliceChannel -gt 0))
    }
    $summary | ConvertTo-Json | Set-Content ([System.IO.Path]::Combine($destination, 'summary.json'))
    $summary | ConvertTo-Json
    if (-not $summary.passed) { throw 'Sensor-link acceptance failed' }
} finally {
    foreach ($writer in $writers.Values) { $writer.Dispose() }
    foreach ($port in $ports.Values) { if ($port.IsOpen) { $port.Close() }; $port.Dispose() }
}
