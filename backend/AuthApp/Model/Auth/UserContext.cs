using Microsoft.EntityFrameworkCore;

namespace SmartStartBack.Model.Auth
{
    public class UserContext : DbContext
    {
        public DbSet<User> Users { get; init; }

        public UserContext()
        {
            //Database.Migrate();
        }

        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
#if DEBUG
            optionsBuilder.UseSqlServer("Data Source=(localdb)\\MSSQLLocalDB; Database=AuthDb");
#else
            optionsBuilder.UseSqlServer("Server=tcp:localhost,1433;UID=sa;PWD=zhenyasenkoSql72;TrustServerCertificate=True");
#endif
        }
    }
}


