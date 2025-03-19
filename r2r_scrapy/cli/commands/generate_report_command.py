import click
import asyncio
import os
import logging
import json
from datetime import datetime
from r2r_scrapy.cli.commands.base_command import BaseCommand
from r2r_scrapy.exporters.r2r_exporter import R2RExporter

class GenerateReportCommand(BaseCommand):
    """Command for generating a quality report"""
    
    async def _get_collections(self):
        """Get all collections from R2R"""
        r2r_settings = self.get_r2r_api_settings()
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        await exporter.initialize()
        
        try:
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
            self.logger.error(f"Error getting collections: {e}")
            return []
        finally:
            await exporter.close()
    
    async def _get_documents(self, collection_id=None, limit=1000):
        """Get documents from R2R"""
        r2r_settings = self.get_r2r_api_settings()
        exporter = R2RExporter(
            api_url=r2r_settings['api_url'],
            api_key=r2r_settings['api_key'],
            batch_size=r2r_settings['batch_size'],
            max_concurrency=r2r_settings['max_concurrency']
        )
        await exporter.initialize()
        
        try:
            url = f"{r2r_settings['api_url']}/documents?limit={limit}"
            if collection_id:
                url += f"&collection_id={collection_id}"
            
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
            self.logger.error(f"Error getting documents: {e}")
            return []
        finally:
            await exporter.close()
    
    def _generate_html_report(self, collections, documents):
        """Generate HTML report"""
        total_collections = len(collections)
        total_documents = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        
        doc_types = {}
        for doc in documents:
            doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        chunking_strategies = {}
        for doc in documents:
            strategy = doc.get('metadata', {}).get('chunking_strategy', 'unknown')
            chunking_strategies[strategy] = chunking_strategies.get(strategy, 0) + 1
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>R2R Scrapy Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .summary {{ background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .chart {{ height: 200px; margin-bottom: 20px; }}
            </style>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>R2R Scrapy Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Collections: {total_collections}</p>
                <p>Total Documents: {total_documents}</p>
                <p>Total Chunks: {total_chunks}</p>
                <p>Average Chunks per Document: {total_chunks / total_documents if total_documents else 0:.2f}</p>
            </div>
            
            <h2>Document Types</h2>
            <div class="chart">
                <canvas id="docTypesChart"></canvas>
            </div>
            
            <h2>Chunking Strategies</h2>
            <div class="chart">
                <canvas id="chunkingChart"></canvas>
            </div>
            
            <h2>Collections</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Documents</th>
                    <th>Created At</th>
                </tr>
        """
        
        for collection in collections:
            html += f"""
                <tr>
                    <td>{collection.get('collection_id', '')}</td>
                    <td>{collection.get('metadata', {}).get('library_name', '')}</td>
                    <td>{collection.get('document_count', 0)}</td>
                    <td>{collection.get('created_at', '')}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Recent Documents</h2>
            <table>
                <tr>
                    <th>ID</th>
                    <th>Title</th>
                    <th>Collection</th>
                    <th>Chunks</th>
                    <th>Type</th>
                </tr>
        """
        
        for doc in documents[:20]:
            title = doc.get('metadata', {}).get('title', '') or doc.get('document_id', '')
            if len(title) > 40:
                title = title[:37] + "..."
            
            html += f"""
                <tr>
                    <td>{doc.get('document_id', '')}</td>
                    <td>{title}</td>
                    <td>{doc.get('collection_id', '')}</td>
                    <td>{doc.get('chunk_count', 0)}</td>
                    <td>{doc.get('metadata', {}).get('doc_type', 'unknown')}</td>
                </tr>
            """
        
        html += f"""
            </table>
            
            <script>
                var docTypesCtx = document.getElementById('docTypesChart').getContext('2d');
                var docTypesChart = new Chart(docTypesCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {json.dumps(list(doc_types.keys()))},
                        datasets: [{{
                            data: {json.dumps(list(doc_types.values()))},
                            backgroundColor: [
                                '#4e73df', '#1cc88a', '#36b9cc', '#f6c23e', '#e74a3b',
                                '#858796', '#5a5c69', '#6610f2', '#6f42c1', '#fd7e14'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        legend: {{ position: 'right' }}
                    }}
                }});
                
                var chunkingCtx = document.getElementById('chunkingChart').getContext('2d');
                var chunkingChart = new Chart(chunkingCtx, {{
                    type: 'pie',
                    data: {{
                        labels: {json.dumps(list(chunking_strategies.keys()))},
                        datasets: [{{
                            data: {json.dumps(list(chunking_strategies.values()))},
                            backgroundColor: [
                                '#1cc88a', '#4e73df', '#36b9cc', '#f6c23e', '#e74a3b',
                                '#858796', '#5a5c69', '#6610f2', '#6f42c1', '#fd7e14'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        legend: {{ position: 'right' }}
                    }}
                }});
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _generate_json_report(self, collections, documents):
        """Generate JSON report"""
        total_collections = len(collections)
        total_documents = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)
        
        doc_types = {}
        for doc in documents:
            doc_type = doc.get('metadata', {}).get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        chunking_strategies = {}
        for doc in documents:
            strategy = doc.get('metadata', {}).get('chunking_strategy', 'unknown')
            chunking_strategies[strategy] = chunking_strategies.get(strategy, 0) + 1
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_collections': total_collections,
                'total_documents': total_documents,
                'total_chunks': total_chunks,
                'avg_chunks_per_document': total_chunks / total_documents if total_documents else 0,
            },
            'document_types': doc_types,
            'chunking_strategies': chunking_strategies,
            'collections': collections,
            'recent_documents': documents[:20],
        }
        
        return json.dumps(report, indent=2)
    
    def run(self, output_format='html', output_file=None, collection_id=None):
        """Run the generate report command"""
        self.logger.info("Generating quality report")
        
        loop = asyncio.get_event_loop()
        collections = loop.run_until_complete(self._get_collections())
        documents = loop.run_until_complete(self._get_documents(collection_id))
        
        documents.sort(key=lambda d: d.get('created_at', ''), reverse=True)
        
        if output_format == 'html':
            report = self._generate_html_report(collections, documents)
        else:
            report = self._generate_json_report(collections, documents)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            click.echo(f"Report saved to: {output_file}")
        else:
            click.echo(report)

@click.command()
@click.option('--format', 'output_format', type=click.Choice(['html', 'json']), 
              default='html', help='Output format')
@click.option('--output', 'output_file', help='Output file path')
@click.option('--collection', 'collection_id', help='Filter by collection ID')
@click.option('--config', help='Path to configuration file')
def generate_report(output_format, output_file, collection_id, config):
    """Generate a quality report"""
    command = GenerateReportCommand(config_path=config)
    command.run(
        output_format=output_format,
        output_file=output_file,
        collection_id=collection_id
    ) 