import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class ListCollectionsCommand(BaseCommand):
    """Command for listing R2R collections"""
    
    async def _list_collections(self):
        """List collections in R2R"""
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
            # Call R2R API to list collections
            async with exporter.session.get(
                f"{r2r_settings['api_url']}/collections",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return []
                
                result = await response.json()
                return result.get('collections', [])
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, output_format='table'):
        """Run the list collections command"""
        self.logger.info("Listing R2R collections")
        
        # Get collections
        loop = asyncio.get_event_loop()
        collections = loop.run_until_complete(self._list_collections())
        
        # Display collections
        if not collections:
            click.echo("No collections found")
            return
        
        if output_format == 'json':
            import json
            click.echo(json.dumps(collections, indent=2))
        else:
            # Display as table
            click.echo("\nCollections:")
            click.echo("-" * 80)
            click.echo(f"{'ID':<30} {'Name':<20} {'Document Count':<15} {'Created At'}")
            click.echo("-" * 80)
            
            for collection in collections:
                click.echo(f"{collection.get('collection_id', ''):<30} "
                           f"{collection.get('metadata', {}).get('library_name', ''):<20} "
                           f"{collection.get('document_count', 0):<15} "
                           f"{collection.get('created_at', '')}")

@click.command()
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--config', help='Path to configuration file')
def list_collections(output_format, config):
    """List R2R collections"""
    command = ListCollectionsCommand(config_path=config)
    command.run(output_format=output_format) 