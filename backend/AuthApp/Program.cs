using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Authorization;
using Microsoft.IdentityModel.Tokens;
using SmartStartBack.Model.Auth;
using SmartStartBack.Model.Path;
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Text;

var builder = WebApplication.CreateBuilder(args);

#region
/*
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme).AddJwtBearer(options=>
{
    options.TokenValidationParameters = new TokenValidationParameters
    {
        // указывает, будет ли валидироваться издатель при валидации токена
        ValidateIssuer = true,
        // строка, представляющая издателя
        ValidIssuer = AuthOptions.ISSUER,
        // будет ли валидироваться потребитель токена
        ValidateAudience = true,
        // установка потребителя токена
        ValidAudience = AuthOptions.AUDIENCE,
        // будет ли валидироваться время существования
        ValidateLifetime = true,
        // установка ключа безопасности
        IssuerSigningKey = AuthOptions.GetSymmetricSecurityKey(),
        // валидация ключа безопасности
        ValidateIssuerSigningKey = true,
    };
}
*/
#endregion

//builder.Services.AddAuthentication(CookieAuthenticationDefaults.AuthenticationScheme).AddCookie();
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme).AddJwtBearer(
    options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidIssuer = AuthOptions.ISSUER,
            ValidateAudience = true,
            ValidAudience = AuthOptions.AUDIENCE,
            ValidateLifetime = true,
            IssuerSigningKey = AuthOptions.GetSymmetricSecurityKey(),
            ValidateIssuerSigningKey = true
        };
    });
builder.Services.AddAuthorization();


builder.Services.AddControllers();
builder.Services.AddSingleton<UserContext>();
builder.Services.AddScoped<AuthManager>();
builder.Services.AddScoped<IPathFinder, DullPathFinder>();
builder.Services.AddHttpClient();

var app = builder.Build();

app.UseAuthentication();
app.UseAuthorization();
app.MapControllers();
app.MapGet("/", () => "Hello World!");
app.MapGet("/edick",(HttpContext context) => {
    var user = context.User.Identity;
    if (user is not null && user.IsAuthenticated) {
        string? answer = context.User.FindFirst(ClaimTypes.Name)?.Value;
        return Results.Ok("ebal ego: " + answer + " "+ user.AuthenticationType);
    }
    else
    {
        return Results.Unauthorized();
    }
});

app.Run();


