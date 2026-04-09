from django import forms


class CsvUploadForm(forms.Form):
    csv_file = forms.FileField(label="CSV 파일")