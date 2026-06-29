from __future__ import annotations


def create_case(query: str, module: str, expected_fields: dict) -> dict:
    return {
        "query": query,
        "expected_module": module,
        "expected_fields": expected_fields,
    }


def delete_case(query: str, module: str, expected_fields: dict) -> dict:
    d = dict(expected_fields)
    d["state"] = "absent"
    return {
        "query": query,
        "expected_module": module,
        "expected_fields": d,
    }


def info_case(query: str, module: str, expected_fields: dict) -> dict:
    return {
        "query": query,
        "expected_module": module,
        "expected_fields": expected_fields,
    }


TESTS = []

# 12 create cases
TESTS.extend([
    create_case(
        "create a virtual network called prod-vnet in resource group prod-rg",
        "azure_rm_virtualnetwork",
        {"name": "prod-vnet", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a subnet called backend-subnet in virtual network prod-vnet resource group prod-rg",
        "azure_rm_subnet",
        {"name": "backend-subnet", "resource_group": "prod-rg", "virtual_network_name": "prod-vnet"},
    ),
    create_case(
        "create a storage account named mystorage in resource group prod-rg",
        "azure_rm_storageaccount",
        {"name": "mystorage", "resource_group": "prod-rg"},
    ),
    create_case(
        "create an AKS cluster named prod-aks in resource group prod-rg",
        "azure_rm_aks",
        {"name": "prod-aks", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a network interface named web-nic in resource group prod-rg",
        "azure_rm_networkinterface",
        {"name": "web-nic", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a virtual network gateway named gateway1 in resource group prod-rg",
        "azure_rm_virtualnetworkgateway",
        {"name": "gateway1", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a route table named route-table1 in resource group prod-rg",
        "azure_rm_routetable",
        {"name": "route-table1", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a public IP address named publicip1 in resource group prod-rg",
        "azure_rm_publicipaddress",
        {"name": "publicip1", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a load balancer named prod-lb in resource group prod-rg",
        "azure_rm_loadbalancer",
        {"name": "prod-lb", "resource_group": "prod-rg"},
    ),
    create_case(
        "create an application gateway named appgw1 in resource group prod-rg",
        "azure_rm_appgateway",
        {"name": "appgw1", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a managed disk named disk1 in resource group prod-rg",
        "azure_rm_manageddisk",
        {"name": "disk1", "resource_group": "prod-rg"},
    ),
    create_case(
        "create a key vault named vault1 in resource group prod-rg",
        "azure_rm_keyvault",
        {"name": "vault1", "resource_group": "prod-rg"},
    ),
])

# 12 delete cases
TESTS.extend([
    delete_case(
        "delete virtual network prod-vnet from resource group prod-rg",
        "azure_rm_virtualnetwork",
        {"name": "prod-vnet", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete subnet backend-subnet from virtual network prod-vnet resource group prod-rg",
        "azure_rm_subnet",
        {"name": "backend-subnet", "resource_group": "prod-rg", "virtual_network_name": "prod-vnet"},
    ),
    delete_case(
        "delete storage account mystorage from resource group prod-rg",
        "azure_rm_storageaccount",
        {"name": "mystorage", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete AKS cluster prod-aks from resource group prod-rg",
        "azure_rm_aks",
        {"name": "prod-aks", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete network interface web-nic from resource group prod-rg",
        "azure_rm_networkinterface",
        {"name": "web-nic", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete virtual network gateway gateway1 from resource group prod-rg",
        "azure_rm_virtualnetworkgateway",
        {"name": "gateway1", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete route table route-table1 from resource group prod-rg",
        "azure_rm_routetable",
        {"name": "route-table1", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete public IP address publicip1 from resource group prod-rg",
        "azure_rm_publicipaddress",
        {"name": "publicip1", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete load balancer prod-lb from resource group prod-rg",
        "azure_rm_loadbalancer",
        {"name": "prod-lb", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete application gateway appgw1 from resource group prod-rg",
        "azure_rm_appgateway",
        {"name": "appgw1", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete managed disk disk1 from resource group prod-rg",
        "azure_rm_manageddisk",
        {"name": "disk1", "resource_group": "prod-rg"},
    ),
    delete_case(
        "delete key vault vault1 from resource group prod-rg",
        "azure_rm_keyvault",
        {"name": "vault1", "resource_group": "prod-rg"},
    ),
])

# 16 info/list cases
TESTS.extend([
    info_case(
        "show details of virtual network prod-vnet",
        "azure_rm_virtualnetwork_info",
        {"name": "prod-vnet"},
    ),
    info_case(
        "list subnets in virtual network prod-vnet",
        "azure_rm_subnet_info",
        {"virtual_network_name": "prod-vnet"},
    ),
    info_case(
        "list storage accounts in resource group prod-rg",
        "azure_rm_storageaccount_info",
        {"resource_group": "prod-rg"},
    ),
    info_case(
        "get information about AKS cluster prod-aks",
        "azure_rm_aks_info",
        {"name": "prod-aks"},
    ),
    info_case(
        "show details of network interface web-nic",
        "azure_rm_networkinterface_info",
        {"name": "web-nic"},
    ),
    info_case(
        "show details of virtual network gateway gateway1",
        "azure_rm_virtualnetworkgateway_info",
        {"name": "gateway1"},
    ),
    info_case(
        "show details of route table route-table1",
        "azure_rm_routetable_info",
        {"name": "route-table1"},
    ),
    info_case(
        "show details of public IP address publicip1",
        "azure_rm_publicipaddress_info",
        {"name": "publicip1"},
    ),
    info_case(
        "show details of load balancer prod-lb",
        "azure_rm_loadbalancer_info",
        {"name": "prod-lb"},
    ),
    info_case(
        "show details of application gateway appgw1",
        "azure_rm_appgateway_info",
        {"name": "appgw1"},
    ),
    info_case(
        "show details of managed disk disk1",
        "azure_rm_manageddisk_info",
        {"name": "disk1"},
    ),
    info_case(
        "show details of key vault vault1",
        "azure_rm_keyvault_info",
        {"name": "vault1"},
    ),
    info_case(
        "show details of web app prod-webapp",
        "azure_rm_webapp_info",
        {"name": "prod-webapp"},
    ),
    info_case(
        "show details of SQL server sqlserver1",
        "azure_rm_sqlserver_info",
        {"name": "sqlserver1"},
    ),
    info_case(
        "show details of SQL database inventorydb",
        "azure_rm_sqldatabase_info",
        {"name": "inventorydb"},
    ),
    info_case(
        "show details of container registry prodregistry",
        "azure_rm_containerregistry_info",
        {"name": "prodregistry"},
    ),
])

assert len(TESTS) == 40, len(TESTS)