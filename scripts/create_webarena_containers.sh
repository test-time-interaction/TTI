#!/bin/bash

# Get server hostname
SERVER_HOSTNAME=""  # You can replace this with your actual server hostname or make it a parameter

# Check if arguments are provided
if [ $# -lt 1 ]; then
    echo "Error: Please provide at least one type:num pair."
    echo "Usage: $0 <type1:num1> [<type2:num2> ...]"
    echo "  <type>: 'reddit', 'shopping', 'shopping_admin', 'gitlab', or 'map'"
    echo "  <num>: number of containers to create"
    echo "Example: $0 reddit:2 shopping:3 gitlab:1"
    exit 1
fi

# Validate input arguments
for arg in "$@"; do
    # Split argument by colon
    type=$(echo $arg | cut -d: -f1)
    num=$(echo $arg | cut -d: -f2)
    
    # Validate type
    if [[ ! "$type" =~ ^(reddit|shopping|shopping_admin|gitlab|map)$ ]]; then
        echo "Error: Type '$type' must be one of: 'reddit', 'shopping', 'shopping_admin', 'gitlab', or 'map'."
        exit 1
    fi
    
    # Validate num
    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        echo "Error: Number for type '$type' must be a positive integer, got '$num'."
        exit 1
    fi
done

# Get all container IDs from docker ps
container_ids=$(docker ps -q -a)

# Check if we got any container IDs
if [ -n "$container_ids" ]; then
  echo "Found container IDs: $container_ids"
  echo "Killing containers..."
  docker kill $container_ids
  echo "Containers killed successfully."
  docker rm $container_ids
else
  echo "No running containers found."
fi
sleep 2

# Function to start containers based on type and number
start_containers() {
    local type=$1
    local num=$2
    
    case "$type" in
        "reddit")
            # Loop from 0 to num-1
            for i in $(seq 0 $((num - 1))); do
                port_num=$((9999 + i))
                container_name="reddit-$i"
                
                echo "Starting reddit container '$container_name' on port $port_num..."
                docker run --name "$container_name" -p $port_num:80 -d postmill-populated-exposed-withimg
                
                if [ $? -eq 0 ]; then
                    echo "Container '$container_name' started successfully on port $port_num."
                else
                    echo "Failed to start container '$container_name'."
                fi
            done
            ;;
            
        "shopping")
            # Loop from 0 to num-1
            for i in $(seq 0 $((num - 1))); do
                port_num=$((7770 - i))
                container_name="shopping-$i"
                
                echo "Starting shopping container '$container_name' on port $port_num..."
                docker run --name "$container_name" -p $port_num:80 -d shopping_final_0712
                
                if [ $? -eq 0 ]; then
                    echo "Container '$container_name' started successfully on port $port_num."                
                    sleep 5
                    
                    echo "Setting base URL..."
                    docker exec "$container_name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$SERVER_HOSTNAME:$port_num"
                    
                    echo "Updating secure base URL in database..."
                    docker exec "$container_name" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$SERVER_HOSTNAME:$port_num/' WHERE path = 'web/secure/base_url';"
                    
                    echo "Flushing cache..."
                    docker exec "$container_name" /var/www/magento2/bin/magento cache:flush
                    
                    echo "Setup completed for container '$container_name'."
                else
                    echo "Failed to start container '$container_name'."
                fi
            done
            ;;
            
        "shopping_admin")
            # Loop from 0 to num-1
            for i in $(seq 0 $((num - 1))); do
                port_num=$((7780 + i))
                container_name="shopping_admin-$i"
                
                echo "Starting shopping_admin container '$container_name' on port $port_num..."
                docker run --name "$container_name" -p $port_num:80 -d shopping_admin_final_0719
                
                if [ $? -eq 0 ]; then
                    echo "Container '$container_name' started successfully on port $port_num."                
                    sleep 5
                    
                    echo "Setting base URL..."
                    docker exec "$container_name" /var/www/magento2/bin/magento setup:store-config:set --base-url="http://$SERVER_HOSTNAME:$port_num"
                    
                    echo "Waiting for configuration to apply..."
                    sleep 5
                    
                    echo "Updating secure base URL in database..."
                    docker exec "$container_name" mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value='http://$SERVER_HOSTNAME:$port_num/' WHERE path = 'web/secure/base_url';"
                    
                    echo "Waiting for database update to apply..."
                    sleep 5
                    
                    echo "Flushing cache..."
                    docker exec "$container_name" /var/www/magento2/bin/magento cache:flush
                    
                    echo "Setup completed for container '$container_name'."
                else
                    echo "Failed to start container '$container_name'."
                fi
            done
            ;;
            
        "gitlab")
            # Loop from 0 to num-1
            for i in $(seq 0 $((num - 1))); do
                port_num=$((8023 + i))
                container_name="gitlab-$i"
                
                echo "Starting gitlab container '$container_name' on port $port_num..."
                docker run --name "$container_name" -d -p $port_num:$port_num gitlab-populated-final-port8023 /opt/gitlab/embedded/bin/runsvdir-start
                
                if [ $? -eq 0 ]; then
                    echo "Container '$container_name' started successfully on port $port_num."
                    
                    echo "Waiting at least 5 minutes for services to boot..."
                    sleep 5
                    
                    echo "Configuring external URL..."
                    docker exec "$container_name" sed -i "s|^external_url.*|external_url 'http://$SERVER_HOSTNAME:$port_num'|" /etc/gitlab/gitlab.rb
                    
                    echo "Reconfiguring GitLab..."
                    docker exec "$container_name" gitlab-ctl reconfigure
                    
                    echo "Setup completed for container '$container_name'."
                else
                    echo "Failed to start container '$container_name'."
                fi
            done
            ;;
            
        "map")
            # Loop from 0 to num-1
            for i in $(seq 0 $((num - 1))); do
                port_num=$((8080 + i))
                container_name="map-$i"
                
                echo "Starting map container '$container_name' on port $port_num..."
                docker run --name "$container_name" -p $port_num:80 -d map-populated-exposed-withimg
                
                if [ $? -eq 0 ]; then
                    echo "Container '$container_name' started successfully on port $port_num."
                else
                    echo "Failed to start container '$container_name'."
                fi
            done
            ;;
    esac
}

# Process each type:num pair
echo "Starting containers for multiple domains..."
for arg in "$@"; do
    # Split argument by colon
    type=$(echo $arg | cut -d: -f1)
    num=$(echo $arg | cut -d: -f2)
    
    echo "Processing $num containers of type '$type'..."
    start_containers "$type" "$num"
done

echo "Completed starting containers for all specified domains."