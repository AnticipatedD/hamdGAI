terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "agent_rg" {
  name     = "rg-hamdgai-production"
  location = "East US"
  
  tags = {
    Environment = "Production"
    ManagedBy   = "Terraform"
    Project     = "hamdGAI"
  }
}
terraform {
  required_version = ">= 1.3.0"
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2.0"
    }
  }
}

resource "null_resource" "hamdgai_cluster_stub" {
  triggers = {
    agent_version = "1.0.0"
  }
}

output "cluster_status" {
  value = "hamdGAI Infrastructure Module Initialized"
}
