using Microsoft.AspNetCore.Identity;
using System.ComponentModel.DataAnnotations;

namespace SmartStartBack.Model.Auth
{
    public class User
    {
        public int Id { get; set; }
        [Required]
        [EmailAddress]
        public string Email { get; set; }
        [Required]
        public string Password { get; set; }
        public string? AuthKey { get; set; }
        public User(string login, string password)
        {
            Email = login;
            Password = password;
        }

        public User() { }
    }
}
