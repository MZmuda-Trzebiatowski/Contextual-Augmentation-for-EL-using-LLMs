# Datasets

This folder contains all used datasets
`/jsons` contains all datasets in json form with schema:

```json
[
    {
        "corpus": "<text>",
        "tags": [
            {
                "text": "<relevant text>",
                "beginIndex": <index>,
                "endIndex": <index>,
                "uri": "<entity uri>"
            },
            ...
        ]
    },
    ...
]    
```
