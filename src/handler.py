#!/usr/bin/env python3
import runpod
import boto3
import os
import logging
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import torch

# ヘッドレス環境用の設定
os.environ["DISPLAY"] = ""  # ディスプレイを無効化
os.environ["OPEN3D_CPU_RENDERING"] = "true"  # CPUレンダリングを有効化
os.environ["PYOPENGL_PLATFORM"] = "egl"  # EGLプラットフォームを使用

import open3d as o3d
import trimesh

# Open3Dをヘッドレスモードで初期化
def init_open3d_headless():
    """Open3Dをヘッドレスモードで初期化"""
    try:
        # ヘッドレスレンダリングを有効化
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Open3D: {e}")
        return False


# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# デバッグ用：利用可能な環境変数を表示
logger.info('Available environment variables:')
for key, value in os.environ.items():
    if any(keyword in key for keyword in ['INPUT', 'OUTPUT', 'R2', 'BUCKET']):
        logger.info(f'{key}: {value}')

class GPUWorker:
    def __init__(self):
        # Open3Dをヘッドレスモードで初期化
        if not init_open3d_headless():
            logger.warning("Open3D headless initialization failed, but continuing...")
        
        # Cloudflare R2の設定
        self.s3_client = boto3.client(
            's3',
            endpoint_url=os.environ.get('R2_ENDPOINT'),
            region_name='auto',
            aws_access_key_id=os.environ.get('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.environ.get('R2_SECRET_ACCESS_KEY'),
            config=boto3.session.Config(signature_version='s3v4')
        )
        
        # 環境変数から設定を読み込み
        self.bucket_name = os.environ.get('R2_BUCKET_NAME')
        
        logger.info(f"Configuration:")
        logger.info(f"- Bucket: {self.bucket_name}")
        logger.info(f"- Endpoint: {os.environ.get('R2_ENDPOINT')}")

    async def test_connection(self):
        """S3接続をテスト"""
        try:
            logger.info("Testing S3 connection...")
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                MaxKeys=5
            )
            
            logger.info("Connection successful!")
            object_count = len(response.get('Contents', []))
            logger.info(f"Found {object_count} objects in bucket")
            
            for obj in response.get('Contents', []):
                logger.info(f"- {obj['Key']} ({obj['Size']} bytes)")
            
            return True
        except Exception as error:
            logger.error(f"Connection test failed: {error}")
            return False

    async def download_file(self, bucket, key, local_path):
        """S3からファイルをダウンロード"""
        logger.info(f"Attempting to download: {key} from bucket: {bucket}")
        
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Successfully downloaded {key} to {local_path}")
        except Exception as error:
            logger.error(f"Failed to download {key}: {error}")
            raise error

    async def upload_file(self, bucket, key, local_path):
        """S3にファイルをアップロード"""
        logger.info(f"Uploading {key} to bucket: {bucket}")
        
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Successfully uploaded {key}")
        except Exception as error:
            logger.error(f"Failed to upload {key}: {error}")
            raise error

    async def run_splat_mesh(self, input_path, output_path):
        logger.info(f"Running splat-mesh: {input_path} -> {output_path}")
        
        try:
            # DN-Splatter の import は環境に合わせて
            # from dn_splatter import mesh_reconstruction

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            # Open3D で読み込み
            logger.info("Loading point cloud...")
            pcd = o3d.io.read_point_cloud(input_path)
            
            if len(pcd.points) == 0:
                raise ValueError("Point cloud is empty or could not be loaded")
            
            logger.info(f"Loaded point cloud with {len(pcd.points)} points")
            
            # 法線を推定
            logger.info("Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            # Poisson Reconstruction 例
            logger.info("Performing Poisson reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            if len(mesh.vertices) == 0:
                raise ValueError("Failed to generate mesh from point cloud")
            
            logger.info(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
            
            # メッシュを保存
            success = o3d.io.write_triangle_mesh(output_path, mesh)
            if not success:
                raise ValueError("Failed to write mesh to file")
                
            logger.info(f"Mesh saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error in run_splat_mesh: {e}")
            raise e


    async def check_gpu_info(self):
        """GPU情報を確認"""
        logger.info("\n=== GPU Information ===")
        
        # nvidia-smi実行（存在する場合）
        try:
            result = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"[NVIDIA-SMI] {stdout}")
            else:
                logger.info(f"nvidia-smi exit code: {result.returncode}")
        except Exception as error:
            logger.info(f"nvidia-smi not available: {error}")
        
        # Vulkan情報の確認
        try:
            result = await asyncio.create_subprocess_exec(
                "vulkaninfo",
                "--summary",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                logger.info(f"[VULKAN] {stdout}")
            else:
                logger.info(f"vulkaninfo exit code: {result.returncode}")
        except Exception as error:
            logger.info(f"vulkaninfo not available: {error}")
        
        logger.info("========================\n")


async def handler(job):
    """RunPodサーバーレス関数のハンドラー（ストリーミング対応版）"""
    try:
        # job_id の取り出し
        job_id = job["id"]
        logger.info(f"Job ID: {job_id}")

        job_input = job.get("input", {})
        logger.info(f"Received job input: {job_input}")
        
        # 初期レスポンス
        yield {
            "status": "started",
            "message": "Job processing started",
            "progress": 0
        }
        
        # 入力から必要なパラメータを取得
        input_key = job_input.get("input_key") or os.environ.get('INPUT_KEY')
        output_key = job_input.get("output_key") or os.environ.get('OUTPUT_KEY')
        # GPUを使用しないフラグ (比較テスト用)
        no_gpu = job_input.get("no_gpu", False)
        
        if not input_key or not output_key:
            yield {
                "status": "error",
                "message": "input_key and output_key are required",
                "progress": -1
            }
            return
        
        # ワーカーインスタンスを作成
        worker = GPUWorker()
        
        yield {
            "status": "initializing",
            "message": "Worker initialized, testing connection...",
            "progress": 10
        }
        
        # 接続テストを実行
        # logger.info("Testing connection to R2...")
        # connection_ok = await worker.test_connection()
        # if not connection_ok:
        #     yield {
        #         "status": "error",
        #         "message": "Failed to connect to R2",
        #         "progress": -1
        #     }
        #     return
        
        yield {
            "status": "downloading",
            "message": f"Downloading file: {input_key}",
            "progress": 20
        }
        
        # 一時ファイルのパスを設定
        # tmp_input = f"/tmp/{Path(input_key).name}"
        # tmp_output = f"/tmp/{Path(output_key).name}"
        tmp_input = f"/tmp/input.ply"
        tmp_output = f"/tmp/output.ply"
        
        # ファイルをダウンロード
        logger.info("Downloading file from R2...")
        await worker.download_file(worker.bucket_name, input_key, tmp_input)
        
        # ダウンロードしたファイルの確認
        input_stats = Path(tmp_input).stat()
        logger.info(f"Downloaded file size: {input_stats.st_size} bytes")
        
        yield {
            "status": "processing",
            "message": f"Running splat-transform (file size: {input_stats.st_size} bytes)",
            "progress": 40,
            "input_size": input_stats.st_size
        }
        
        # splat-transformを実行
        logger.info("Running splat-transform...")
        await worker.run_splat_mesh(tmp_input, tmp_output)

        # 変換後のファイルの確認
        output_stats = Path(tmp_output).stat()
        logger.info(f"Converted file size: {output_stats.st_size} bytes")


        # TODO: texture を剥がすかどうか検討 Option引数

        # glb 形式に
        tmp_input = tmp_output
        tmp_output = f"/tmp/output.glb"
        mesh = trimesh.load(tmp_input)
        mesh.export(tmp_output)

        yield {
            "status": "uploading",
            "message": f"Uploading result: {output_key}",
            "progress": 80,
            "output_size": output_stats.st_size
        }
        
        # 結果をアップロード
        logger.info("Uploading result to R2...")
        await worker.upload_file(worker.bucket_name, output_key, tmp_output)
        
        # 一時ファイルの削除
        Path(tmp_input).unlink(missing_ok=True)
        Path(tmp_output).unlink(missing_ok=True)
        logger.info("Temporary files cleaned up")
        
        # 最終結果
        yield {
            "status": "completed",
            "message": "Job completed successfully",
            "progress": 100,
            "input_file": input_key,
            "output_file": output_key,
            "input_size": input_stats.st_size,
            "output_size": output_stats.st_size
        }
        
        logger.info("Job completed successfully!")
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        
        # エラー時も一時ファイルを削除
        try:
            Path(tmp_input).unlink(missing_ok=True)
            Path(tmp_output).unlink(missing_ok=True)
        except:
            pass
        
        yield {
            "status": "error",
            "message": f"Handler error: {str(e)}",
            "progress": -1
        }

# RunPodサーバーレス関数として開始（ストリーミング対応）
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True  # ストリーミングレスポンスを有効化
})
