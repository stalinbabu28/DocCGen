from pipeline.select_document import select_document

TESTS = [

    ("create a virtual network", "azure_rm_virtualnetwork"),
    ("create a subnet", "azure_rm_subnet"),
    ("create a storage account", "azure_rm_storageaccount"),
    ("create a virtual machine", "azure_rm_virtualmachine"),
    ("create an AKS cluster", "azure_rm_aks"),

    ("create a network interface", "azure_rm_networkinterface"),
    ("create a virtual network gateway", "azure_rm_virtualnetworkgateway"),
    ("create a route table", "azure_rm_routetable"),
    ("create a public IP address", "azure_rm_publicipaddress"),
    ("create a load balancer", "azure_rm_loadbalancer"),

    ("create an application gateway", "azure_rm_appgateway"),
    ("create a managed disk", "azure_rm_manageddisk"),
    ("create a key vault", "azure_rm_keyvault"),
    ("create a resource group", "azure_rm_resourcegroup"),
    ("create a DNS zone", "azure_rm_dnszone"),

    ("create a virtual machine scale set", "azure_rm_virtualmachinescaleset"),
    ("create a web app", "azure_rm_webapp"),
    ("create a SQL server", "azure_rm_sqlserver"),
    ("create a SQL database", "azure_rm_sqldatabase"),
    ("create a container registry", "azure_rm_containerregistry"),

    ("list storage accounts", "azure_rm_storageaccount_info"),
    ("show details of a virtual network", "azure_rm_virtualnetwork_info"),
    ("list subnets", "azure_rm_subnet_info"),
    ("get AKS cluster information", "azure_rm_aks_info")
]

correct = 0

for query, expected in TESTS:

    selected = select_document(query)

    module = (
        selected
        .split("/")[-1]
        .replace("azure.azcollection.", "")
        .replace(".json", "")
    )

    print("\n" + "=" * 80)
    print("Query:", query)
    print("Expected:", expected)
    print("Selected:", module)

    if module == expected:
        print("CORRECT")
        correct += 1
    else:
        print("WRONG")

accuracy = correct / len(TESTS)

print("\n" + "=" * 80)
print("Selector Accuracy:", round(accuracy * 100, 2), "%")