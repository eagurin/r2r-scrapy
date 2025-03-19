import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class DeleteCollectionCommand(BaseCommand):
    """Command for deleting an R2R collection"""
    
    async def _delete_collection(self, collection_id):
        """Delete a collection from R2R"""
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
            # Call R2R API to delete collection
            async with exporter.session.delete(
                f"{r2r_settings['api_url']}/collections/{collection_id}",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status not in (200, 204):
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return {'error': error_text}
                
                return {'success': True}
        except Exception as e:
            self.logger.error(f"Error deleting collection: {e}")
            return {'error': str(e)}
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, collection_id, force=False):
        """Run the delete collection command"""
        self.logger.info(f"Deleting collection: {collection_id}")
        
        # Confirm deletion if not forced
        if not force:
            confirm = click.confirm(f"Are you sure you want to delete collection '{collection_id}'?")
            if not confirm:
                click.echo("Deletion cancelled")
                return
        
        # Delete collection
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._delete_collection(collection_id))
        
        # Display result
        if 'error' in result:
            click.echo(f"Error deleting collection: {result['error']}")
        else:
            click.echo(f"Collection deleted: {collection_id}")

@click.command()
@click.option('--id', 'collection_id', required=True, help='Collection ID to delete')
@click.option('--force', is_flag=True, help='Delete without confirmation')
@click.option('--config', help='Path to configuration file')
def delete_collection(collection_id, force, config):
    """Delete an R2R collection"""
    command = DeleteCollectionCommand(config_path=config)
    command.run(collection_id=collection_id, force=force) 