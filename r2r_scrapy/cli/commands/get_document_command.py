import click
import asyncio
import os
import logging
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class GetDocumentCommand(BaseCommand):
    """Command for getting a document from R2R"""
    
    async def _get_document(self, document_id):
        """Get a document from R2R"""
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
            # Call R2R API to get document
            async with exporter.session.get(
                f"{r2r_settings['api_url']}/documents/{document_id}",
                headers={
                    "Authorization": f"Bearer {r2r_settings['api_key']}",
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    self.logger.error(f"R2R API error: {response.status} - {error_text}")
                    return None
                
                result = await response.json()
                return result
        except Exception as e:
            self.logger.error(f"Error getting document: {e}")
            return None
        finally:
            # Close exporter
            await exporter.close()
    
    def run(self, document_id, output_format='json', output_file=None, include_chunks=False):
        """Run the get document command"""
        self.logger.info(f"Getting document: {document_id}")
        
        # Get document
        loop = asyncio.get_event_loop()
        document = loop.run_until_complete(self._get_document(document_id))
        
        # Check if document exists
        if not document:
            click.echo(f"Document not found: {document_id}")
            return
        
        # Format output
        if output_format == 'json':
            import json
            output = json.dumps(document, indent=2)
        elif output_format == 'text':
            # Just output the content
            output = document.get('content', '')
        else:  # summary
            # Output a summary
            metadata = document.get('metadata', {})
            title = metadata.get('title', '') or document_id
            collection_id = document.get('collection_id', '')
            chunk_count = document.get('chunk_count', 0)
            content_preview = document.get('content', '')[:200] + '...' if document.get('content') else ''
            
            output = f"Document: {document_id}\n"
            output += f"Title: {title}\n"
            output += f"Collection: {collection_id}\n"
            output += f"Chunks: {chunk_count}\n"
            output += f"Metadata: {json.dumps(metadata, indent=2)}\n"
            output += f"Content Preview:\n{content_preview}\n"
        
        # Output result
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(output)
            click.echo(f"Document saved to: {output_file}")
        else:
            click.echo(output)

@click.command()
@click.option('--id', 'document_id', required=True, help='Document ID to get')
@click.option('--format', 'output_format', type=click.Choice(['json', 'text', 'summary']), 
              default='json', help='Output format')
@click.option('--output', 'output_file', help='Output file path')
@click.option('--include-chunks', is_flag=True, help='Include document chunks')
@click.option('--config', help='Path to configuration file')
def get_document(document_id, output_format, output_file, include_chunks, config):
    """Get a document from R2R"""
    command = GetDocumentCommand(config_path=config)
    command.run(
        document_id=document_id,
        output_format=output_format,
        output_file=output_file,
        include_chunks=include_chunks
    ) 