param(
    [Parameter(Mandatory = $true)][string]$TemplatePath,
    [Parameter(Mandatory = $true)][string]$OutputJson,
    [Parameter(Mandatory = $true)][string]$OutputPdf
)

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($TemplatePath, $false, $true)

    $paragraphs = @()
    for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
        $p = $doc.Paragraphs.Item($i)
        $text = ($p.Range.Text -replace "[\r\a]", "").Trim()
        $paragraphs += [ordered]@{
            index = $i
            text = $text
            style = [string]$p.Range.Style.NameLocal
            alignment = $p.Alignment
            font_name = [string]$p.Range.Font.Name
            font_size = $p.Range.Font.Size
            bold = $p.Range.Font.Bold
            first_line_indent = $p.Format.FirstLineIndent
            left_indent = $p.Format.LeftIndent
            space_before = $p.Format.SpaceBefore
            space_after = $p.Format.SpaceAfter
            line_spacing = $p.Format.LineSpacing
        }
    }

    $tables = @()
    for ($ti = 1; $ti -le $doc.Tables.Count; $ti++) {
        $table = $doc.Tables.Item($ti)
        $rows = @()
        for ($ri = 1; $ri -le $table.Rows.Count; $ri++) {
            $cells = @()
            for ($ci = 1; $ci -le $table.Rows.Item($ri).Cells.Count; $ci++) {
                $cell = $table.Rows.Item($ri).Cells.Item($ci)
                $cells += (($cell.Range.Text -replace "[\r\a]", "").Trim())
            }
            $rows += ,$cells
        }
        $tables += [ordered]@{
            index = $ti
            rows = $table.Rows.Count
            columns = $table.Columns.Count
            content = $rows
        }
    }

    $sections = @()
    for ($si = 1; $si -le $doc.Sections.Count; $si++) {
        $section = $doc.Sections.Item($si)
        $setup = $section.PageSetup
        $sections += [ordered]@{
            index = $si
            page_width = $setup.PageWidth
            page_height = $setup.PageHeight
            orientation = $setup.Orientation
            top_margin = $setup.TopMargin
            bottom_margin = $setup.BottomMargin
            left_margin = $setup.LeftMargin
            right_margin = $setup.RightMargin
            header_distance = $setup.HeaderDistance
            footer_distance = $setup.FooterDistance
            header_text = (($section.Headers.Item(1).Range.Text -replace "[\r\a]", "").Trim())
            footer_text = (($section.Footers.Item(1).Range.Text -replace "[\r\a]", "").Trim())
        }
    }

    $result = [ordered]@{
        path = $TemplatePath
        page_count = $doc.ComputeStatistics(2)
        paragraph_count = $doc.Paragraphs.Count
        table_count = $doc.Tables.Count
        section_count = $doc.Sections.Count
        paragraphs = $paragraphs
        tables = $tables
        sections = $sections
    }
    $result | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $OutputJson -Encoding UTF8
    $doc.ExportAsFixedFormat($OutputPdf, 17)
}
finally {
    if ($doc -ne $null) { $doc.Close(0) }
    if ($word -ne $null) { $word.Quit() }
    if ($doc -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc) }
    if ($word -ne $null) { [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
