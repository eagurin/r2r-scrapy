import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class CreateCollectionCommand(BaseCommand):
    """Command for creating a new R2R collection"""
    
    async def _create_collection(self, collection_id, name, description, metadata=None):
        """Create a new collection in R2R"""
        # Get R2R API settings
        r2r_settings = self.get_r2r_api_settings()
        
        # Create R2R exporter
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        
        # Initialize exporter
        await exporter.initialize()
        
        try:
            # Prepare collection metadata
            collection_metadata = metadata or {}
            collection_metadata.update({
                'name': name,
                'description': description,
            })
            
            # Create collection
            result = await exporter.create_collection(collection_id, collection_metadata)
            return result
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return {'error': str(e)}
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, name, collection_id=None, description=None, metadata=None):
        """Run the create collection command"""
        # Generate collection ID if not provided
        if not collection_id:
            collection_id = name.lower().replace(' ', '_')
        
        self.logger.info(f"Creating collection: {name} (ID: {collection_id})")
        
        # Create collection
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._create_collection(
            collection_id=collection_id,
            name=name,
            description=description or f"Collection for {name}",
            metadata=metadata
        ))
        
        # Display result
        if 'error' in result:
            click.echo(f"Error creating collection: {result['error']}")
        else:
            click.echo(f"Collection created: {collection_id}")
            click.echo(f"Result: {result}")

@click.command()
@click.option('--name', required=True, help='Collection name')
@click.option('--id', 'collection_id', help='Collection ID (generated from name if not provided)')
@click.option('--description', help='Collection description')
@click.option('--metadata', help='JSON string with additional metadata')
@click.option('--config', help='Path to configuration file')
def create_collection(name, collection_id, description, metadata, config):
    """Create a new R2R collection"""
    # Parse metadata JSON if provided
    metadata_dict = None
    if metadata:
        import json
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            click.echo("Error: Metadata must be a valid JSON string")
            return
    
    command = CreateCollectionCommand(config_path=config)
    command.run(
        name=name,
        collection_id=collection_id,
        description=description,
        metadata=metadata_dict
    ) 