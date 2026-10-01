"""Real optional workbook decoding retains selected sheets, identifiers and blank rows."""
import hashlib
from zipfile import ZipFile

import pytest
import wrangle as wr


def workbook(path):
    files = {
        '[Content_Types].xml': '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/><Override PartName="/xl/worksheets/sheet2.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>',
        '_rels/.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>',
        'xl/workbook.xml': '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Control" sheetId="1" r:id="rId1"/><sheet name="Measurements" sheetId="2" r:id="rId2"/></sheets></workbook>',
        'xl/_rels/workbook.xml.rels': '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/></Relationships>',
        'xl/worksheets/sheet1.xml': '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>different</t></is></c></row><row r="2"><c r="A2"><v>999</v></c></row></sheetData></worksheet>',
        'xl/worksheets/sheet2.xml': '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="A1:B4"/><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>id</t></is></c><c r="B1" t="inlineStr"><is><t>mass</t></is></c></row><row r="2"><c r="A2" t="inlineStr"><is><t>001</t></is></c><c r="B2"><v>1.5</v></c></row><row r="3"/><row r="4"><c r="A4" t="inlineStr"><is><t>002</t></is></c><c r="B4"><v>2.5</v></c></row></sheetData></worksheet>',
    }
    with ZipFile(path, 'w') as output:
        for name, content in files.items():
            output.writestr(name, content)


def test_actual_selected_workbook_preserves_identifiers_and_explicit_spacer_exclusion(tmp_path):
    pytest.importorskip('fastexcel')
    path = tmp_path / 'instrument.xlsx'
    workbook(path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {'path': path, 'format': 'excel', 'options': {'sheet_name': 'Measurements'}, 'schema': {'id': 'String', 'mass': 'Float64'}}
    profile = wr.inspect(source)
    assert profile['rows'] == 3
    assert profile['examples'] == [{'id': '001', 'mass': 1.5}, {'id': None, 'mass': None}, {'id': '002', 'mass': 2.5}]
    with pytest.raises(wr.WrangleError) as error:
        wr.prepare(source, {'key': 'id'})
    assert error.value.code == 'MISSING_REQUIRED'
    result = wr.prepare(source, {'units': {'mass': 'mg'}, 'steps': [{'op': 'filter', 'where': {'is_not_null': {'col': 'id'}}, 'reason': 'Workbook spacer rows have no measurement', 'key': 'id'}]})
    assert result.data['id'].to_list() == ['001', '002']
    assert result.receipt['units'] == {'mass': 'mg'}
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum
    source['options'] = {'sheet_id': 2}
    assert wr.inspect(source)['examples'] == profile['examples']


def test_excel_cli_failure_remains_one_structured_error(tmp_path, capsys):
    import json
    from wrangle._cli import main
    pytest.importorskip('fastexcel')
    path = tmp_path / 'instrument.xlsx'
    workbook(path)
    recipe = tmp_path / 'protocol.yaml'
    from wrangle._protocol import dump_recipe
    recipe.write_text(dump_recipe({'key': 'id', 'source_options': {'data': {'format': 'excel', 'options': {'sheet_name': 'Measurements'}, 'schema': {'id': 'String', 'mass': 'Float64'}}}}))
    assert main(['prepare', str(recipe), '--json', '--source', f'data={path}']) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert json.loads(captured.err)['code'] == 'MISSING_REQUIRED'
