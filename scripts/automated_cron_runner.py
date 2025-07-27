# scripts/automated_cron_runner.py
import asyncio
import sys
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.cloud_settings import cloud_config
from scripts.cloud_production_pipeline import CloudProductionPipeline

logger = logging.getLogger(__name__)

class AutomatedCronRunner:
    """
    Automated cron runner for cloud deployment
    Runs every minute, checks market status, executes pipeline if market is open
    """
    
    def __init__(self):
        self.config = cloud_config
        self.pipeline = None
        self.last_run_time = None
        self.run_stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'market_closed_skips': 0,
            'errors': 0,
            'last_error': None
        }
        
    async def should_run_pipeline(self) -> dict:
        """Check if pipeline should run (market open + time interval)"""
        try:
            if not self.pipeline:
                self.pipeline = CloudProductionPipeline()
            
            # Check market status
            market_status = await self.pipeline.check_market_status()
            
            if not market_status.get('is_market_open'):
                return {
                    'should_run': False,
                    'reason': 'market_closed',
                    'market_status': market_status
                }
            
            # Check if enough time has passed since last run
            cycle_interval = self.config.TRADING_CONFIG['cycle_interval_minutes']
            
            if self.last_run_time:
                time_since_last = (datetime.now() - self.last_run_time).total_seconds() / 60
                if time_since_last < cycle_interval:
                    return {
                        'should_run': False,
                        'reason': 'too_soon',
                        'time_since_last_minutes': time_since_last,
                        'interval_minutes': cycle_interval
                    }
            
            return {
                'should_run': True,
                'market_status': market_status
            }
            
        except Exception as e:
            logger.error(f"Error checking if pipeline should run: {e}")
            return {
                'should_run': False,
                'reason': 'error',
                'error': str(e)
            }
    
    async def run_pipeline_cycle(self) -> dict:
        """Run one pipeline cycle"""
        try:
            self.run_stats['total_runs'] += 1
            
            if not self.pipeline:
                self.pipeline = CloudProductionPipeline()
            
            # Run the pipeline
            result = await self.pipeline.run_cloud_pipeline_cycle()
            
            if result.get('cycle_completed'):
                self.run_stats['successful_runs'] += 1
                self.last_run_time = datetime.now()
                
                # Log success
                logger.info(f"Pipeline cycle completed successfully")
                
                return {
                    'success': True,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                reason = result.get('reason', 'unknown')
                if reason == 'market_closed':
                    self.run_stats['market_closed_skips'] += 1
                else:
                    self.run_stats['errors'] += 1
                    self.run_stats['last_error'] = reason
                
                return {
                    'success': False,
                    'reason': reason,
                    'result': result,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            self.run_stats['errors'] += 1
            self.run_stats['last_error'] = str(e)
            logger.error(f"Error in pipeline cycle: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cron_tick(self) -> dict:
        """Single cron tick - called every minute"""
        tick_start = datetime.now()
        
        try:
            # Check if we should run
            should_run_check = await self.should_run_pipeline()
            
            if not should_run_check['should_run']:
                # Save status for monitoring
                status = {
                    'tick_time': tick_start.isoformat(),
                    'action': 'skipped',
                    'reason': should_run_check['reason'],
                    'stats': self.run_stats.copy()
                }
                
                # Log skip reason
                reason = should_run_check['reason']
                if reason == 'market_closed':
                    logger.info("Market closed - skipping pipeline run")
                elif reason == 'too_soon':
                    logger.info(f"Too soon since last run - skipping (last run: {self.last_run_time})")
                
                return status
            
            # Run pipeline
            logger.info("Market is open - running pipeline cycle")
            cycle_result = await self.run_pipeline_cycle()
            
            duration = (datetime.now() - tick_start).total_seconds()
            
            status = {
                'tick_time': tick_start.isoformat(),
                'action': 'executed',
                'duration_seconds': duration,
                'cycle_result': cycle_result,
                'stats': self.run_stats.copy()
            }
            
            # Save status to file for dashboard
            await self._save_status(status)
            
            return status
            
        except Exception as e:
            logger.error(f"Error in cron tick: {e}")
            
            return {
                'tick_time': tick_start.isoformat(),
                'action': 'error',
                'error': str(e),
                'stats': self.run_stats.copy()
            }
    
    async def _save_status(self, status: dict):
        """Save current status for dashboard monitoring"""
        try:
            # Save to logs directory
            status_file = self.config.LOGS_DIR / 'current_status.json'
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
            
            # Also save timestamped version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            history_file = self.config.LOGS_DIR / f'status_history_{timestamp}.json'
            with open(history_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Error saving status: {e}")


# dashboard/dashboard_server.py
import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from aiohttp import web, web_ws
import aiohttp_cors
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.cloud_settings import cloud_config

logger = logging.getLogger(__name__)

class AthenaeDashboard:
    """
    Real-time dashboard for monitoring Athena pipeline
    Shows live data, unusual flow alerts, and system status
    """
    
    def __init__(self):
        self.config = cloud_config
        self.connected_clients = set()
        
    async def websocket_handler(self, request):
        """WebSocket handler for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.connected_clients.add(ws)
        logger.info("Dashboard client connected")
        
        try:
            # Send initial data
            await self._send_initial_data(ws)
            
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    if msg.data == 'close':
                        await ws.close()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
                    
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            self.connected_clients.discard(ws)
            logger.info("Dashboard client disconnected")
        
        return ws
    
    async def _send_initial_data(self, ws):
        """Send initial dashboard data to client"""
        try:
            data = await self._get_dashboard_data()
            await ws.send_str(json.dumps({
                'type': 'initial_data',
                'data': data
            }))
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    async def _get_dashboard_data(self) -> dict:
        """Compile dashboard data from various sources"""
        try:
            # Get current status
            status_file = self.config.LOGS_DIR / 'current_status.json'
            current_status = {}
            if status_file.exists():
                with open(status_file, 'r') as f:
                    current_status = json.load(f)
            
            # Get recent snapshots
            snapshots_dir = self.config.DATA_DIR / 'bronze' / 'market_snapshots'
            recent_snapshots = []
            
            if snapshots_dir.exists():
                today = datetime.now().strftime('%Y%m%d')
                today_dir = snapshots_dir / today
                
                if today_dir.exists():
                    snapshot_files = sorted(today_dir.glob('*.json'), 
                                          key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    for file in snapshot_files[:5]:  # Last 5 snapshots
                        try:
                            with open(file, 'r') as f:
                                snapshot = json.load(f)
                                recent_snapshots.append(snapshot.get('summary', {}))
                        except:
                            continue
            
            # Get unusual flow data
            flow_dir = self.config.DATA_DIR / 'bronze' / 'options_flow'
            unusual_flow = []
            
            if flow_dir.exists():
                today = datetime.now().strftime('%Y%m%d')
                today_flow_dir = flow_dir / today
                
                if today_flow_dir.exists():
                    flow_files = sorted(today_flow_dir.glob('*.json'),
                                      key=lambda x: x.stat().st_mtime, reverse=True)
                    
                    for file in flow_files[:3]:  # Last 3 flow files
                        try:
                            with open(file, 'r') as f:
                                flow_data = json.load(f)
                                unusual_flow.extend([
                                    {
                                        'symbol': symbol,
                                        'score': data.get('analysis', {}).get('unusual_activity_score', 0),
                                        'timestamp': data.get('collection_time')
                                    }
                                    for symbol, data in flow_data.items()
                                    if data.get('analysis', {}).get('unusual_activity', False)
                                ])
                        except:
                            continue
            
            # Get system stats
            symbols_count = len(self.config.get_symbol_list())
            
            return {
                'current_status': current_status,
                'recent_snapshots': recent_snapshots,
                'unusual_flow': unusual_flow[:10],  # Top 10
                'system_info': {
                    'environment': self.config.environment,
                    'symbols_monitored': symbols_count,
                    'data_directories': {
                        'bronze_files': len(list(self.config.DATA_DIR.glob('bronze/**/*.json'))) if self.config.DATA_DIR.exists() else 0,
                        'silver_files': len(list(self.config.DATA_DIR.glob('silver/**/*.parquet'))) if self.config.DATA_DIR.exists() else 0
                    }
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def broadcast_update(self, data):
        """Broadcast update to all connected clients"""
        if not self.connected_clients:
            return
        
        message = json.dumps({
            'type': 'update',
            'data': data
        })
        
        # Send to all clients
        disconnected = set()
        for ws in self.connected_clients:
            try:
                await ws.send_str(message)
            except Exception as e:
                logger.warning(f"Error sending to client: {e}")
                disconnected.add(ws)
        
        # Remove disconnected clients
        self.connected_clients -= disconnected
    
    async def api_status(self, request):
        """API endpoint for current status"""
        data = await self._get_dashboard_data()
        return web.json_response(data)
    
    async def api_symbols(self, request):
        """API endpoint for symbol list"""
        symbols = self.config.get_symbol_list()
        return web.json_response({
            'symbols': symbols,
            'count': len(symbols),
            'timestamp': datetime.now().isoformat()
        })
    
    async def api_unusual_flow(self, request):
        """API endpoint for unusual flow data"""
        try:
            flow_dir = self.config.DATA_DIR / 'bronze' / 'options_flow'
            today = datetime.now().strftime('%Y%m%d')
            today_flow_dir = flow_dir / today
            
            all_unusual = []
            
            if today_flow_dir.exists():
                flow_files = sorted(today_flow_dir.glob('*.json'),
                                  key=lambda x: x.stat().st_mtime, reverse=True)
                
                for file in flow_files[:5]:  # Last 5 files
                    try:
                        with open(file, 'r') as f:
                            flow_data = json.load(f)
                            
                            for symbol, data in flow_data.items():
                                analysis = data.get('analysis', {})
                                if analysis.get('unusual_activity', False):
                                    all_unusual.append({
                                        'symbol': symbol,
                                        'timestamp': data.get('collection_time'),
                                        'score': analysis.get('unusual_activity_score', 0),
                                        'total_premium': analysis.get('total_premium', 0),
                                        'patterns': analysis.get('patterns', {}),
                                        'alerts': data.get('alerts', [])
                                    })
                    except:
                        continue
            
            # Sort by score, descending
            all_unusual.sort(key=lambda x: x['score'], reverse=True)
            
            return web.json_response({
                'unusual_flow': all_unusual[:20],  # Top 20
                'total_found': len(all_unusual),
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return web.json_response({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def serve_dashboard(self, request):
        """Serve the main dashboard HTML"""
        html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Athena Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(0,0,0,0.2);
            padding: 20px;
            text-align: center;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #00f5ff, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .container {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 20px;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .card {
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .card h2 {
            margin-bottom: 15px;
            color: #00f5ff;
            border-bottom: 2px solid #00f5ff;
            padding-bottom: 5px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active { background: #4ade80; }
        .status-inactive { background: #ef4444; }
        .status-warning { background: #f59e0b; }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .metric:last-child { border-bottom: none; }
        
        .metric-value {
            font-weight: bold;
            color: #00f5ff;
        }
        
        .unusual-flow-item {
            background: rgba(255,107,107,0.2);
            border-left: 4px solid #ff6b6b;
            padding: 10px;
            margin: 8px 0;
            border-radius: 5px;
        }
        
        .flow-symbol {
            font-weight: bold;
            font-size: 1.1em;
            color: #ff6b6b;
        }
        
        .flow-score {
            float: right;
            background: #ff6b6b;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
        }
        
        .timestamp {
            font-size: 0.8em;
            color: rgba(255,255,255,0.7);
            margin-top: 5px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: rgba(255,255,255,0.7);
        }
        
        .error {
            background: rgba(239,68,68,0.2);
            border: 1px solid #ef4444;
            color: #fecaca;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🏛️ ATHENA</h1>
        <p>Real-time Trading Signal Detection & Options Flow Analysis</p>
    </div>
    
    <div class="container">
        <!-- System Status -->
        <div class="card">
            <h2>📊 System Status</h2>
            <div id="system-status" class="loading">Loading...</div>
        </div>
        
        <!-- Market Data -->
        <div class="card">
            <h2>📈 Market Data</h2>
            <div id="market-data" class="loading">Loading...</div>
        </div>
        
        <!-- Pipeline Stats -->
        <div class="card">
            <h2>⚡ Pipeline Stats</h2>
            <div id="pipeline-stats" class="loading">Loading...</div>
        </div>
        
        <!-- Unusual Options Flow -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>🚨 Unusual Options Flow</h2>
            <div id="unusual-flow" class="loading">Loading...</div>
        </div>
        
        <!-- Recent Activity -->
        <div class="card" style="grid-column: 1 / -1;">
            <h2>📋 Recent Activity</h2>
            <div id="recent-activity" class="loading">Loading...</div>
        </div>
    </div>
    
    <script>
        class AthenaDashboard {
            constructor() {
                this.ws = null;
                this.reconnectDelay = 1000;
                this.maxReconnectDelay = 30000;
                this.connect();
                this.startPeriodicUpdates();
            }
            
            connect() {
                try {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                    
                    this.ws.onopen = () => {
                        console.log('Connected to Athena Dashboard');
                        this.reconnectDelay = 1000;
                    };
                    
                    this.ws.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    };
                    
                    this.ws.onclose = () => {
                        console.log('Disconnected from dashboard');
                        setTimeout(() => this.connect(), this.reconnectDelay);
                        this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                    };
                    
                } catch (error) {
                    console.error('Error connecting:', error);
                    setTimeout(() => this.connect(), this.reconnectDelay);
                }
            }
            
            handleMessage(message) {
                if (message.type === 'initial_data' || message.type === 'update') {
                    this.updateDashboard(message.data);
                }
            }
            
            updateDashboard(data) {
                this.updateSystemStatus(data);
                this.updateMarketData(data);
                this.updatePipelineStats(data);
                this.updateUnusualFlow(data);
                this.updateRecentActivity(data);
            }
            
            updateSystemStatus(data) {
                const status = data.current_status || {};
                const systemInfo = data.system_info || {};
                
                let html = '';
                
                // Market status
                const isMarketOpen = status.action === 'executed' || 
                                   (status.reason !== 'market_closed' && status.action !== 'skipped');
                
                html += `
                    <div class="metric">
                        <span><span class="status-indicator ${isMarketOpen ? 'status-active' : 'status-inactive'}"></span>Market Status</span>
                        <span class="metric-value">${isMarketOpen ? 'OPEN' : 'CLOSED'}</span>
                    </div>
                    <div class="metric">
                        <span>Environment</span>
                        <span class="metric-value">${systemInfo.environment || 'Unknown'}</span>
                    </div>
                    <div class="metric">
                        <span>Symbols Monitored</span>
                        <span class="metric-value">${systemInfo.symbols_monitored || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Last Update</span>
                        <span class="metric-value">${new Date(status.tick_time || Date.now()).toLocaleTimeString()}</span>
                    </div>
                `;
                
                document.getElementById('system-status').innerHTML = html;
            }
            
            updateMarketData(data) {
                const snapshots = data.recent_snapshots || [];
                
                if (snapshots.length === 0) {
                    document.getElementById('market-data').innerHTML = '<p>No recent market data</p>';
                    return;
                }
                
                const latest = snapshots[0];
                const marketData = latest.market_data || {};
                
                const html = `
                    <div class="metric">
                        <span>Symbols Processed</span>
                        <span class="metric-value">${marketData.successful_collections || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Success Rate</span>
                        <span class="metric-value">${marketData.success_rate || 0}%</span>
                    </div>
                    <div class="metric">
                        <span>Stock Data Points</span>
                        <span class="metric-value">${(marketData.total_stock_data_points || 0).toLocaleString()}</span>
                    </div>
                    <div class="metric">
                        <span>Options Trades</span>
                        <span class="metric-value">${(marketData.total_options_trades || 0).toLocaleString()}</span>
                    </div>
                `;
                
                document.getElementById('market-data').innerHTML = html;
            }
            
            updatePipelineStats(data) {
                const stats = data.current_status?.stats || {};
                
                const html = `
                    <div class="metric">
                        <span>Total Runs</span>
                        <span class="metric-value">${stats.total_runs || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Successful Runs</span>
                        <span class="metric-value">${stats.successful_runs || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Market Closed Skips</span>
                        <span class="metric-value">${stats.market_closed_skips || 0}</span>
                    </div>
                    <div class="metric">
                        <span>Errors</span>
                        <span class="metric-value">${stats.errors || 0}</span>
                    </div>
                `;
                
                document.getElementById('pipeline-stats').innerHTML = html;
            }
            
            updateUnusualFlow(data) {
                const unusualFlow = data.unusual_flow || [];
                
                if (unusualFlow.length === 0) {
                    document.getElementById('unusual-flow').innerHTML = '<p>No unusual options flow detected</p>';
                    return;
                }
                
                let html = '';
                unusualFlow.forEach(flow => {
                    html += `
                        <div class="unusual-flow-item">
                            <div class="flow-symbol">${flow.symbol}<span class="flow-score">${flow.score}</span></div>
                            <div class="timestamp">${new Date(flow.timestamp).toLocaleString()}</div>
                        </div>
                    `;
                });
                
                document.getElementById('unusual-flow').innerHTML = html;
            }
            
            updateRecentActivity(data) {
                const snapshots = data.recent_snapshots || [];
                
                if (snapshots.length === 0) {
                    document.getElementById('recent-activity').innerHTML = '<p>No recent activity</p>';
                    return;
                }
                
                let html = '<div style="max-height: 200px; overflow-y: auto;">';
                snapshots.forEach(snapshot => {
                    const timestamp = new Date(snapshot.timestamp || Date.now()).toLocaleString();
                    const marketData = snapshot.market_data || {};
                    
                    html += `
                        <div class="metric">
                            <span>${timestamp}</span>
                            <span class="metric-value">${marketData.successful_collections || 0} symbols processed</span>
                        </div>
                    `;
                });
                html += '</div>';
                
                document.getElementById('recent-activity').innerHTML = html;
            }
            
            async startPeriodicUpdates() {
                // Fetch data via REST API every 30 seconds as backup
                setInterval(async () => {
                    try {
                        const response = await fetch('/api/status');
                        const data = await response.json();
                        this.updateDashboard(data);
                    } catch (error) {
                        console.error('Error fetching status:', error);
                    }
                }, 30000);
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', () => {
            new AthenaDashboard();
        });
    </script>
</body>
</html>
        '''
        
        return web.Response(text=html_content, content_type='text/html')


async def create_dashboard_app():
    """Create dashboard web application"""
    app = web.Application()
    dashboard = AthenaeDashboard()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Routes
    app.router.add_get('/', dashboard.serve_dashboard)
    app.router.add_get('/ws', dashboard.websocket_handler)
    app.router.add_get('/api/status', dashboard.api_status)
    app.router.add_get('/api/symbols', dashboard.api_symbols)
    app.router.add_get('/api/unusual-flow', dashboard.api_unusual_flow)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app


async def main_cron():
    """Main cron runner - runs every minute"""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(cloud_config.LOGS_DIR / 'cron_runner.log'),
            logging.StreamHandler()
        ]
    )
    
    logger.info("🤖 Starting Athena automated cron runner")
    
    try:
        runner = AutomatedCronRunner()
        result = await runner.cron_tick()
        
        # Print result for cron logging
        print(json.dumps(result, indent=2, default=str))
        
        # Exit with status code
        if result.get('action') == 'error':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Cron runner failed: {e}")
        print(json.dumps({'error': str(e), 'timestamp': datetime.now().isoformat()}))
        sys.exit(1)


async def main_dashboard():
    """Main dashboard server"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    app = await create_dashboard_app()
    
    port = int(os.getenv('DASHBOARD_PORT', 8080))
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    
    logger.info(f"📊 Athena Dashboard running on port {port}")
    logger.info(f"🌐 Access at: http://localhost:{port}")
    
    try:
        await asyncio.Future()  # Run forever
    except KeyboardInterrupt:
        logger.info("Dashboard stopped")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'dashboard':
        asyncio.run(main_dashboard())
    else:
        asyncio.run(main_cron())