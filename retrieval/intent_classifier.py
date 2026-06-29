def classify_intent(query):

    query = query.lower()

    resource_words = [
        "create",
        "deploy",
        "provision",
        "update",
        "modify",
        "delete",
        "remove"
    ]

    info_words = [
        "show",
        "get",
        "list",
        "display",
        "describe",
        "information",
        "details"
    ]

    for word in resource_words:
        if word in query:
            return "resource"

    for word in info_words:
        if word in query:
            return "info"

    return "unknown"