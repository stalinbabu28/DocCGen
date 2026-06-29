import os

from retrieval.intent_classifier import classify_intent


def rerank(query, results):

    intent = classify_intent(query)

    query_lower = query.lower()

    reranked = []

    for score, path in results:

        adjusted_score = score

        filename = os.path.basename(path)

        module_name = (
            filename
            .replace("azure.azcollection.", "")
            .replace(".json", "")
        )

        is_info_module = "_info" in module_name

        # ---------------------------------
        # Intent adjustment
        # ---------------------------------

        if intent == "resource":

            if is_info_module:
                adjusted_score -= 0.15

        elif intent == "info":

            if is_info_module:
                adjusted_score += 0.15

        # ---------------------------------
        # Module-name matching bonus
        # ---------------------------------

        module_tokens = (
            module_name
            .replace("azure_rm_", "")
            .replace("_info", "")
            .replace("_", " ")
            .lower()
        )

        query_tokens = query_lower.split()

        matches = 0

        for token in query_tokens:

            if token in module_tokens:
                matches += 1

        adjusted_score += matches * 0.05

        # ---------------------------------
        # Generic penalties
        # ---------------------------------

        if "instance" in module_tokens:
            adjusted_score -= 0.05

        if "extension" in module_tokens:
            adjusted_score -= 0.05

        if "link" in module_tokens:
            adjusted_score -= 0.05

        if "group" in module_tokens:
            adjusted_score -= 0.05

        if (
            "devtestlab" in module_tokens
            and "devtestlab" not in query_lower
        ):
            adjusted_score -= 0.10

        if (
            "privateendpoint" in module_tokens
            and "private endpoint" not in query_lower
        ):
            adjusted_score -= 0.05

        # ---------------------------------
        # DNS-specific adjustment
        # ---------------------------------

        if "dns zone" in query_lower:

            if module_name == "azure_rm_dnszone":
                adjusted_score += 0.10

            elif "privatednszonelink" in module_name:
                adjusted_score -= 0.15

            elif "privatednszone" in module_name:
                adjusted_score -= 0.10

        # ---------------------------------
        # Key Vault adjustment
        # ---------------------------------

        if "key vault" in query_lower:

            if module_name == "azure_rm_keyvault":
                adjusted_score += 0.10

            elif (
                "keyvaultkey" in module_name
                or "keyvaultsecret" in module_name
            ):
                adjusted_score -= 0.10

        reranked.append(
            (adjusted_score, path)
        )

    reranked.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return reranked