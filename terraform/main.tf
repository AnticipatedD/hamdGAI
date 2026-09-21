terraform {
  required_version = ">= 1.0.0"
}

module "agent_cluster" {
  source = "./modules/agent_cluster"
}
