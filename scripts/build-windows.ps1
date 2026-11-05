$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $root
try {
    $entry = New-Item -ItemType Directory -Force -Path 'build/windows-entry'
    Set-Content -Encoding ascii -LiteralPath (Join-Path $entry.FullName 'cli.py') -Value @(
        'from opendft.cli import main', 'raise SystemExit(main())'
    )
    Set-Content -Encoding ascii -LiteralPath (Join-Path $entry.FullName 'gui.py') -Value @(
        'from opendft.gui import main', 'main()'
    )
    foreach ($mode in @('CLI', 'GUI')) {
        $subsystem = if ($mode -eq 'GUI') { '--windowed' } else { '--console' }
        $script = Join-Path $entry.FullName ($mode.ToLowerInvariant() + '.py')
        python -m PyInstaller --noconfirm --onefile $subsystem --paths src `
            --name "YoungCC_for_Windows_$mode" $script
        if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed for $mode." }
    }
} finally {
    Pop-Location
}
