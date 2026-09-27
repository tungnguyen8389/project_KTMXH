# Config package initialization

# Register PyMySQL as the MySQLdb driver so Django's django.db.backends.mysql
# works without building the C-based mysqlclient. Harmless when SQLite is used.
try:
    import pymysql
    pymysql.install_as_MySQLdb()
except ImportError:
    pass
