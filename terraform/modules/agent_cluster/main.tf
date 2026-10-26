resource "terraform_data" "cluster" {
  input = {
    cluster_name = var.cluster_name
  }
}
