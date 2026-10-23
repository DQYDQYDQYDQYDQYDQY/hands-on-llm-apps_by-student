"""小型 SQLite 检查点与产物库；事务保存状态，正文不塞入事件日志。"""
import hashlib,json,re,sqlite3
from pathlib import Path

def encode(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))

class CheckpointConflict(RuntimeError):
    pass

class CheckpointStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink(): raise ValueError('课堂数据库不能是符号链接')
        self.connection=sqlite3.connect(self.path)
        self.connection.execute('CREATE TABLE IF NOT EXISTS runs (id TEXT PRIMARY KEY, revision INTEGER, state TEXT)')
        self.connection.execute('CREATE TABLE IF NOT EXISTS artifacts (run_id TEXT, id TEXT, body TEXT, PRIMARY KEY(run_id,id))')
        self.connection.commit()

    def close(self): self.connection.close()

    def create(self, state):
        with self.connection:
            self.connection.execute('INSERT INTO runs VALUES (?,?,?)',(state['run_id'],0,encode(state)))
        return dict(state),0

    def load(self, run_id):
        row=self.connection.execute('SELECT state,revision FROM runs WHERE id=?',(run_id,)).fetchone()
        if row is None: raise KeyError('run 不存在')
        return json.loads(row[0]),row[1]

    def save(self, state, expected_revision):
        # 乐观并发检查：旧读取者不能覆盖另一执行者刚保存的状态。
        with self.connection:
            cursor=self.connection.execute('UPDATE runs SET state=?,revision=revision+1 WHERE id=? AND revision=?',
                (encode(state),state['run_id'],expected_revision))
            if cursor.rowcount!=1: raise CheckpointConflict('状态已改变，请重新读取')
        return expected_revision+1

    def put_artifact(self, run_id, value):
        body=encode(value); artifact_id=hashlib.sha256(body.encode()).hexdigest()
        with self.connection:
            self.connection.execute('INSERT OR IGNORE INTO artifacts VALUES (?,?,?)',(run_id,artifact_id,body))
        return {'id':artifact_id,'size_bytes':len(body.encode()),'sha256':artifact_id}

    def read_artifact(self, run_id, artifact_id):
        if not isinstance(artifact_id,str) or not re.fullmatch('[0-9a-f]{64}',artifact_id):
            raise ValueError('产物 ID 应为 64 位十六进制哈希')
        row=self.connection.execute('SELECT body FROM artifacts WHERE run_id=? AND id=?',(run_id,artifact_id)).fetchone()
        if row is None: raise KeyError('当前 run 无此产物')
        if hashlib.sha256(row[0].encode()).hexdigest()!=artifact_id: raise ValueError('产物内容校验失败')
        return json.loads(row[0])
