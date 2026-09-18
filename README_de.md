# MELD – Anleitung zur Installation und Ausführung

MELD (**Machine Learning Execution and Deployment**) ermöglicht es, ein bereitgestelltes Machine-Learning-Modell mit Daten aus dem **AKTIN Data Warehouse (DWH)** auszuführen.

Für die normale Verwendung müssen Sie das Modell nicht selbst installieren oder konfigurieren. MELD übernimmt die technischen Schritte für die Ausführung.

MELD besteht vereinfacht aus zwei Teilen:

* **MELD Orchestrator:** Bereitet die benötigten Daten aus dem AKTIN-DWH vor, startet die Modellausführung und speichert anschließend Ergebnisse und Protokolle.
* **Modell-Laufzeitumgebung (Inference Runtime):** Enthält das bereitgestellte Modell und alle Programme, die für dessen Ausführung benötigt werden. Sie wird in einem eigenen Docker-Container ausgeführt.

Welches Modell verwendet wird und welche Daten dafür benötigt werden, steht in einer bereitgestellten **Contract-Datei** (`contract.yaml`).

---

# 1. Voraussetzungen

Für die Verwendung von MELD werden benötigt:

1. Ein Linux-System mit **Docker** und **Docker Compose**.
2. Ein installiertes und laufendes **AKTIN-DWH**.
3. Die aktuelle MELD-Datei `compose.yaml`.
4. Die Zugangsdaten für die Datenbank des AKTIN-DWH.
5. Eine bereitgestellte `contract.yaml` für das auszuführende Modell.

## Docker installieren

Docker muss auf dem System installiert sein. Die offizielle Installationsanleitung finden Sie unter:

[Docker installieren](https://docs.docker.com/engine/install/?utm_source=chatgpt.com)

---

# 2. MELD-Verzeichnis vorbereiten

Erstellen Sie einen Ordner, in dem MELD betrieben werden soll.

Öffnen Sie anschließend ein Terminal und wechseln Sie in diesen Ordner.

Beispiel:

```bash
cd /pfad/zum/meld-ordner
```

Alle folgenden Befehle werden in diesem Ordner ausgeführt.

---

# 3. MELD herunterladen bzw. aktualisieren

Laden Sie die aktuelle `compose.yaml` herunter:

```bash
curl -o compose.yaml https://raw.githubusercontent.com/aktin/MELD/refs/heads/main/scripts/compose.yaml
```

Wenn MELD bereits eingerichtet wurde, kann auf diese Weise die vorhandene `compose.yaml` durch die aktuelle Version ersetzt werden.

---

# 4. Verbindung zum AKTIN-DWH einrichten

MELD benötigt die Verbindungsinformationen für das lokale AKTIN-DWH.

Erstellen Sie dazu im gleichen Ordner wie die `compose.yaml` eine Datei mit dem Namen:

```text
.env
```

Tragen Sie dort die Datenbankverbindung ein:

```dotenv
# Verbindung zum AKTIN-DWH
# DB_HOST=my-host
DB_PORT=5432
DB_USER=i2b2crcdata
DB_SCHEMA=i2b2

# Zeitzone
TZ=Europe/Berlin
```

### Hinweis zu `DB_HOST`

Wenn das AKTIN-DWH auf demselben Rechner betrieben wird, soll `DB_HOST` nicht gesetzt werden.

Befindet sich das DWH auf einem anderen Rechner, muss die entsprechende Adresse eingetragen werden.

---

# 5. Datenbankpasswort hinterlegen

Erstellen Sie im gleichen Ordner eine Datei mit dem Namen:

```text
db_password.txt
```

In diese Datei wird **ausschließlich das Passwort der Datenbank** eingetragen.

Beispiel:

```text
mein-datenbankpasswort
```

Alternativ kann die Datei über das Terminal erstellt werden:

```bash
echo "mein-datenbankpasswort" > db_password.txt
```

Ersetzen Sie dabei `mein-datenbankpasswort` durch das tatsächliche Passwort.

---

# 6. MELD starten

Bevor MELD gestartet wird, muss einmalig für das aktuelle Terminal die benötigte Docker-Einstellung gesetzt werden:

```bash
export DOCKER_SOCKET_GID="$(stat -c '%g' /var/run/docker.sock)"
```

Starten Sie anschließend MELD:

```bash
docker compose up -d
```

MELD läuft danach im Hintergrund. Das Terminal kann weiterhin verwendet werden.

Beim ersten Start werden automatisch unter anderem folgende Ordner angelegt:

| Ordner       | Inhalt                                                  |
| ------------ | ------------------------------------------------------- |
| `contracts/` | Contract-Dateien für die Modelle                        |
| `jobs/`      | Ergebnisse und Dateien der einzelnen Modellausführungen |
| `logs/`      | Protokolldateien von MELD                               |

---

# 7. Contract bereitstellen

Sie erhalten für das vorgesehene Modell eine Datei mit dem Namen:

```text
contract.yaml
```

Kopieren Sie diese Datei in:

```text
contracts/
```

Die Datei sollte sich anschließend hier befinden:

```text
contracts/contract.yaml
```

Der Contract beschreibt unter anderem, welches Modell verwendet wird, welche Daten benötigt werden und welche Ergebnisse erwartet werden.

---

# 8. Modell herunterladen

Vor der ersten Ausführung muss die benötigte Modell-Laufzeitumgebung heruntergeladen werden.

Öffnen Sie ein Terminal und wechseln Sie in den Ordner mit der `compose.yaml`.

Führen Sie anschließend aus:

```bash
docker compose exec meld orchestrator pull contract.yaml
```

Docker lädt daraufhin die für den Contract vorgesehene Modell-Laufzeitumgebung herunter.

Je nach Größe des Modells und Geschwindigkeit der Netzwerkverbindung kann dies einige Zeit dauern.

## Modelle aus einer privaten Registry

Dieser zusätzliche Schritt ist **nur erforderlich**, wenn das Modell in einer privaten Container-Registry gespeichert ist.

Melden Sie sich zunächst bei der Registry an:

```bash
docker login <registry>
```

Folgen Sie anschließend den Anweisungen von Docker.

Falls Ihnen stattdessen ein Zugriffstoken zur Verfügung gestellt wurde:

```bash
export DOCKER_TOKEN=<token>
echo $DOCKER_TOKEN | docker login <registry> --username <user> --password-stdin
```

Ersetzen Sie `<registry>`, `<token>` und `<user>` durch die bereitgestellten Angaben.

Weitere Informationen:

[Docker – Anmeldung bei einer Registry](https://docs.docker.com/reference/cli/docker/login/?utm_source=chatgpt.com)

---

# 9. Modell einmalig ausführen

Für eine einzelne Modellausführung führen Sie folgenden Befehl aus:

```bash
docker compose exec meld orchestrator run contract.yaml
```

MELD führt anschließend automatisch die notwendigen Schritte aus:

1. Die benötigten Daten werden aus dem AKTIN-DWH abgefragt.
2. Die Eingabedaten für das Modell werden vorbereitet.
3. Das Modell wird in einer eigenen Laufzeitumgebung gestartet.
4. Die Ergebnisse werden eingesammelt und geprüft.
5. Protokolle und Ergebnisse werden gespeichert.

Für jede Ausführung erstellt MELD einen eigenen Ordner unter:

```text
jobs/
```

Ein solcher Ordner kann beispielsweise so aussehen:

```text
jobs/stationary-admission-test-tfdf_20260708153747/
```

Der Name des erzeugten Job-Ordners wird bei der Ausführung im Terminal angezeigt.

---

# 10. Ergebnisse abrufen

Nach einer erfolgreichen Ausführung befindet sich das Ergebnisarchiv im entsprechenden Job-Ordner:

```text
jobs/<job_id>/output/summarized_execution.zip
```

Beispiel:

```text
jobs/stationary-admission-test-tfdf_20260708153747/output/summarized_execution.zip
```

Diese ZIP-Datei kann aus dem Job-Ordner kopiert und entsprechend dem vorgesehenen Ablauf weitergegeben oder archiviert werden.

Das Archiv enthält:

* `input/` – die für die Ausführung verwendeten Eingabedateien und den Contract,
* `logs/` – die Protokolle der Ausführung,
* `output/` – das vom Modell erzeugte Ergebnis (`output.csv`).

**Wichtig:** Verwenden Sie für die Weitergabe der Ergebnisse das vorgesehene Ergebnisarchiv. Einzelne Dateien innerhalb des Job-Ordners sollten nicht verändert werden.

---

# 11. Automatische Ausführung

Die automatische Ausführung ist derzeit nicht implementiert. Das Paket
`ExecutionScheduler` bleibt als unabhängige Hülle für eine spätere Umsetzung
erhalten.

---

# 12. Modell entfernen

Wenn das Modell nicht mehr benötigt wird, kann die heruntergeladene Modell-Laufzeitumgebung wieder entfernt werden.

Entfernen Sie anschließend das Modell:

```bash
docker compose exec meld orchestrator remove contract.yaml
```

Das Entfernen ist nach einer normalen Modellausführung **nicht erforderlich**. Dieser Schritt dient dazu, nicht mehr benötigte Modell-Laufzeitumgebungen vom System zu entfernen.

---

# Kurzfassung für den regulären Ablauf

## MELD starten

```bash
export DOCKER_SOCKET_GID="$(stat -c '%g' /var/run/docker.sock)"
docker compose up -d
```

## Contract bereitstellen

`contract.yaml` nach folgendem Ordner kopieren:

```text
contracts/
```

## Modell herunterladen

```bash
docker compose exec meld orchestrator pull contract.yaml
```

## Modell einmalig ausführen

```bash
docker compose exec meld orchestrator run contract.yaml
```

## Ergebnis entnehmen

Das Ergebnisarchiv befindet sich unter:

```text
jobs/<job_id>/output/summarized_execution.zip
```

## Modell bei Bedarf entfernen

```bash
docker compose exec meld orchestrator remove contract.yaml
```

# Bei Problemen

Wenn ein Befehl mit einer Fehlermeldung beendet wird:

1. Kopieren oder notieren Sie die **vollständige Fehlermeldung**.
2. Notieren Sie, welcher Befehl ausgeführt wurde.
3. Falls bereits ein Job angelegt wurde, notieren Sie die angezeigte **Job-ID**.
4. Verändern oder löschen Sie den zugehörigen Job-Ordner und die Logdateien nicht.
5. Übermitteln Sie diese Informationen über den vorgesehenen AKTIN-Supportweg.

Enthält die Fehlermeldung möglicherweise Zugangsdaten oder andere vertrauliche Informationen, sollte sie vor der Weitergabe entsprechend geprüft werden.
