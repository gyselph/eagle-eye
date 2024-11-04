# Security features

This folder contains the feature engineering step of EagleEye. We provide here the Python scripts necessary to extract security features from raw behavior data.

## Run the demo feature extraction

- Navigate to the repository root directory
- Use Python 3.12
- Create a pip virtual environment:
```
python -m venv .venv
source ./.venv/bin/activate
```
- Install required Python libraries:
```
pip install -r ./requirements.txt
```
- Run the main script:
```
python ./2-Security-features/main.py
```

This code takes in 1 graph with raw features, and outputs a new graph with security features. The input graph is located under [./input/sample_graph.json](./input/sample_graph.json). The output graph is stored under [./output/graph_with_security_features.json](./output/graph_with_security_features.json). The following 4 security features are computed:
- `encoding_length_command_line`: Compute the length of a command-line for a process event
- `encoding_create_file_disposition`: Compute a categorical feature representation for the file disposition
- `encoding_internet`: A boolean feature which indicates whether a registry key is related to networking settings
- `encoding_ip_source_internal`: A boolean feature which indicates whether a socket source is in the internal network

## Input and output format

Both the input and output of this processing step are process provenance graphs in JSON tree format. You can use a library like `networkx` to import the graphs with Python.

Here is a simplified input graph:

```json
{
    "cmdline": "app.exe -f some-flag",
    "id": "0",
    "children": [
        {
            "event_type": "si_set_value_key",
            "object_name": "\\REGISTRY\\USER\\S-1-5-21-2197411953-2306974247-3794874973-1001\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\IntranetName",
            "id": "1"
        }
    ]
}
```

This graph contains a `registry` event with the registry key as *raw* feature.

As output of the data pipeline, the simplified output graph would look as follows:

```json
{
    "cmdline": "app.exe -f some-flag",
    "id": "0",
    "children": [
        {
            "event_type": "si_set_value_key",
            "object_name": "\\REGISTRY\\USER\\S-1-5-21-2197411953-2306974247-3794874973-1001\\Software\\Microsoft\\Windows\\CurrentVersion\\Internet Settings\\ZoneMap\\IntranetName",
            "encoding_internet": true,
            "id": "1"
        }
    ]
}
```

The output graph contains a security feature *encoding_internet*. This is a boolean feature and indicates whether the registry key is related to networking settings.

## Bring you own data

Note that this step is highly dependent on your particular dataset and its format. This data processing step is limited by the details available in the input - the more *raw* features you can provide, the more *security* features can be extracted.

In this folder, we demonstrate how you can extract *four* specific security features, one per behavior event type. Note that EagleEye uses many more security features. Since your data will likely have a different format than our data, you will need to imlement your own feature extractor. You can use the scripts here as a reference.

## List of security features for EagleEye

Here we provide a list of all security features used for the EagleEye paper. For each security feature, we provide more details about the extraction logic.

### All event types
|Description|Feature name|Feature type|Details|
|---|---|---|---|
|Duration|`encoding_time_span`|Numerical|Duration of behavior event in milliseconds.|

### Process events

These events represent the start of a new process.
|Description|Feature name|Feature type|Details|
|---|---|---|---|
|Name of a binary|`encoding_binary_name`|Categorical|The name of the application that was started. We searched in the training dataset for the 100 most frequently occurring binaries, and created a class for each of them. All other binary names get assigned to a `OTHER` catch-all class.|
|Path of binary|`encoding_process_path`|Categorical|The file path of the binary. We split the whole file system into 16 high-level categories, including system folder, user data, and autostart folder.|
|File extension|`encoding_binary_extension`|Categorical|The file extension of executables, like `.exe`, `.sh`, etc.|
|Dropped binary|`encoding_dropped_binary`|Boolean|Check if a binary was written to disk prior to being started as new process.|
|Lenth of command-line string|`encoding_length_command_line`|Numerical||
|Number of command-line flags|`encoding_number_of_cmdline_args`|Numerical|Count the number of flags present in the command-line string|
|Command-line embedding|`encoding_cmdline_embedding_x`|Numerical|A 16-dimensional vector with encodes the complete command-line string. See step `3-Command-line-embedding` for more details.|

### File system events

|Description|Feature name|Feature type|Details|
|---|---|---|---|
|File path|`encoding_file_path`|Categorical|Same as `encoding_process_path`. All in all, there are 19 path categories.|
|File extension|`encoding_file_extension`|Categorical|The file extension of files being accessed. We searched for the 50 most frequently occurring extensions, and use an `OTHER` catch-all class for the rare extensions.|
|File open options|`encoding_open_options`|Categorical|Read, write, or read-write access.|
|File access flags|`encoding_open_access_flags`|Categorical|Shared or exclusive file access.|
|File disposition|`encoding_create_file_disposition`|Categorical|As per official [Microsoft documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilea), this flag controls the behavior when opening files. Options include file appending, overwriting, or fail if file doesn't exist.|
|Number of bytes accessed|`encoding_file_access_amount`|Numerical|The number of bytes written or read|

### Windows registry access

|Description|Feature name|Feature type|Details|
|---|---|---|---|
|Internet key|`encoding_internet`|Boolean|The registry key is related to internet/networking settings.|
|Persistence key|`encoding_common_persistence`|Categorical|We located 10 keys frequently used by apps to create persistence. These are all well-known persistence keys, such as the `Run` key `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows\CurrentVersion\Run`.|
|Hidden persistence|`encoding_special_persistence`|Categorical|Registry keys can be abused by malware to achieve persistence in an obfuscated way. We located 7 such registry keys, such as `HKEY_LOCAL_MACHINE\Software\Microsoft\Windows NT\CurrentVersion\IniFileMapping\`.|
|Uninstall key|`encoding_unistall`|Boolean|Uninstall an application via registry key. Such keys include `HKEY_LOCAL_MACHINE\Microsoft\Windows\CurrentVersion\Uninstall`.|
|Notify key|`encoding_notify`|Boolean|Create a user notification in graphical form.|
|Data type of key|`encoding_blob_type`|Categorical|Registry values can be of 4 different data types.|
|Root type|`encoding_root`|Categorical|There are 3 registry path roots: User, default user, and machine.|

### Network connections

|Description|Feature name|Feature type|Details|
|---|---|---|---|
|Internal source|`encoding_ip_source_internal`|Boolean|Is the connection source from within the same network?|
|Internal destination|`encoding_ip_destination_internal`|Boolean|Check if the socket destination is in the same network.|
|Service port|`encoding_service_port`|Categorical|We searched for the 20 most frequently used server ports. All other ports are captured in an `OTHER` class.|
|Payload size|`encoding_size_connection`|Numerical|Size of the socket data sent in bytes.|
|Transport layer protocol|`encoding_protocol`|Categorical|TCP or UDP.|
|Incoming connection|`encoding_incoming_connection`|Boolean|Check if the connection is initiated locally or remotely.|

Note: The EagleEye system is benchmarked on two different datasets in the EagleEye paper. The two datasets have different raw features, and thus not all security features are available for each dataset.