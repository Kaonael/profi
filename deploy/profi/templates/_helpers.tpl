{{/*
Expand the name of the chart.
*/}}
{{- define "profi.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a fully-qualified app name.
Truncated at 63 chars because some Kubernetes name fields are limited to this.
*/}}
{{- define "profi.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Chart name + version, used for the app.kubernetes.io/version label.
*/}}
{{- define "profi.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Image reference: repository:tag (tag falls back to Chart.AppVersion)
*/}}
{{- define "profi.image" -}}
{{- $tag := default .Chart.AppVersion .Values.image.tag -}}
{{- printf "%s:%s" .Values.image.repository $tag -}}
{{- end }}

{{/*
Common labels
*/}}
{{- define "profi.labels" -}}
helm.sh/chart: {{ include "profi.chart" . }}
{{ include "profi.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: gpu-profiler
app.kubernetes.io/part-of: profi
{{- end }}

{{/*
Selector labels (must be stable across upgrades — no chart version here)
*/}}
{{- define "profi.selectorLabels" -}}
app.kubernetes.io/name: {{ include "profi.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
ServiceAccount name
*/}}
{{- define "profi.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "profi.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Full runbook URL for a given alert slug.
Usage: {{ include "profi.runbook" (dict "ctx" . "slug" "nccl-hang") }}
*/}}
{{- define "profi.runbook" -}}
{{- printf "%s/%s.md" .ctx.Values.prometheusRule.runbookBaseUrl .slug -}}
{{- end }}
