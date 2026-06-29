from __future__ import annotations

from evaluation.test_cases_40 import TESTS as BASE_TESTS


TESTS = list(BASE_TESTS)

TESTS.extend(
    [
        {
            "query": "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8",
            "expected_module": "azure_rm_virtualnetwork",
            "expected_fields": {
                "name": "prod-vnet",
                "resource_group": "prod-rg",
                "dns_servers": ["1.1.1.1", "8.8.8.8"],
            },
        },
        {
            "query": "create a virtual network called prod-vnet in resource group prod-rg with tags env prod team platform",
            "expected_module": "azure_rm_virtualnetwork",
            "expected_fields": {
                "name": "prod-vnet",
                "resource_group": "prod-rg",
                "tags": {
                    "env": "prod",
                    "team": "platform",
                },
            },
        },
        {
            "query": "create a virtual network called prod-vnet in resource group prod-rg with append tags false",
            "expected_module": "azure_rm_virtualnetwork",
            "expected_fields": {
                "name": "prod-vnet",
                "resource_group": "prod-rg",
                "append_tags": False,
            },
        },
        {
            "query": "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8 and tags env prod team platform",
            "expected_module": "azure_rm_virtualnetwork",
            "expected_fields": {
                "name": "prod-vnet",
                "resource_group": "prod-rg",
                "dns_servers": ["1.1.1.1", "8.8.8.8"],
                "tags": {
                    "env": "prod",
                    "team": "platform",
                },
            },
        },
        {
            "query": "create a virtual network called prod-vnet in resource group prod-rg with dns servers 1.1.1.1 and 8.8.8.8 and tags env prod team platform and append tags true",
            "expected_module": "azure_rm_virtualnetwork",
            "expected_fields": {
                "name": "prod-vnet",
                "resource_group": "prod-rg",
                "dns_servers": ["1.1.1.1", "8.8.8.8"],
                "tags": {
                    "env": "prod",
                    "team": "platform",
                },
                "append_tags": True,
            },
        },
        {
            "query": "create a storage account named mystorage in resource group prod-rg with https only true",
            "expected_module": "azure_rm_storageaccount",
            "expected_fields": {
                "name": "mystorage",
                "resource_group": "prod-rg",
                "https_only": True,
            },
        },
        {
            "query": "create a storage account named mystorage in resource group prod-rg with allow blob public access false",
            "expected_module": "azure_rm_storageaccount",
            "expected_fields": {
                "name": "mystorage",
                "resource_group": "prod-rg",
                "allow_blob_public_access": False,
            },
        },
        {
            "query": "create a storage account named mystorage in resource group prod-rg with kind StorageV2",
            "expected_module": "azure_rm_storageaccount",
            "expected_fields": {
                "name": "mystorage",
                "resource_group": "prod-rg",
                "kind": "StorageV2",
            },
        },
        {
            "query": "create a storage account named mystorage in resource group prod-rg with minimum tls version TLS1_2",
            "expected_module": "azure_rm_storageaccount",
            "expected_fields": {
                "name": "mystorage",
                "resource_group": "prod-rg",
                "minimum_tls_version": "TLS1_2",
            },
        },
        {
            "query": "create a storage account named mystorage in resource group prod-rg with tags env prod costcenter 42",
            "expected_module": "azure_rm_storageaccount",
            "expected_fields": {
                "name": "mystorage",
                "resource_group": "prod-rg",
                "tags": {
                    "env": "prod",
                    "costcenter": "42",
                },
            },
        },
    ]
)

assert len(TESTS) == 50, len(TESTS)