# Generated by Django 4.0.5 on 2023-01-17 19:25

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='meteo_report',
            name='weather',
            field=models.CharField(max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='meteo_report',
            name='wind',
            field=models.FloatField(null=True),
        ),
        migrations.AlterField(
            model_name='culture',
            name='name',
            field=models.CharField(max_length=100, unique=True),
        ),
        migrations.AlterField(
            model_name='meteo_report',
            name='period',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='meteo_report', to='api.period'),
        ),
        migrations.DeleteModel(
            name='Regression_prognoses',
        ),
    ]
