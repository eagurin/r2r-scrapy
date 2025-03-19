import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class ListDocumentsCommand(BaseCommand):
    """Command for listing documents in an R2R collection"""
    
    async def _list_documents(self, collection_id=None, limit=100, offset=0):
        """List documents in R2R"""
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
            # Build URL
            url = f"{r2r_settings['api_url']}/documents?limit={limit}&offset={offset}"
            if collection_id:
                url += f"&collection_id={collection_id}"
            
            # Call R2R API to list documents
            async with exporter.session.get(
                url,
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
                return result.get('documents', [])
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            return []
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, collection_id=None, limit=100, offset=0, output_format='table'):
        """Run the list documents command"""
        self.logger.info(f"Listing documents{' in collection: ' + collection_id if collection_id else ''}")
        
        # Get documents
        loop = asyncio.get_event_loop()
        documents = loop.run_until_complete(self._list_documents(
            collection_id=collection_id,
            limit=limit,
            offset=offset
        ))
        
        # Display documents
        if not documents:
            click.echo("No documents found")
            return
        
        if output_format == 'json':
            import json
            click.echo(json.dumps(documents, indent=2))
        else:
            # Display as table
            click.echo("\nDocuments:")
            click.echo("-" * 100)
            click.echo(f"{'ID':<36} {'Title':<40} {'Collection':<20} {'Chunks'}")
            click.echo("-" * 100)
            
            for doc in documents:
                title = doc.get('metadata', {}).get('title', '') or doc.get('document_id', '')
                if len(title) > 38:
                    title = title[:35] + "..."
                
                click.echo(f"{doc.get('document_id', ''):<36} "
                           f"{title:<40} "
                           f"{doc.get('collection_id', ''):<20} "
                           f"{doc.get('chunk_count', 0)}")

@click.command()
@click.option('--collection', 'collection_id', help='Collection ID to list documents from')
@click.option('--limit', type=int, default=100, help='Maximum number of documents to list')
@click.option('--offset', type=int, default=0, help='Offset for pagination')
@click.option('--format', 'output_format', type=click.Choice(['table', 'json']), 
              default='table', help='Output format')
@click.option('--config', help='Path to configuration file')
def list_documents(collection_id, limit, offset, output_format, config):
    """List documents in an R2R collection"""
    command = ListDocumentsCommand(config_path=config)
    command.run(
        collection_id=collection_id,
        limit=limit,
        offset=offset,
        output_format=output_format
    ) 